from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError as exc:  # pragma: no cover - declared project dependency
    raise SystemExit("psutil is required. Run: python -m pip install -r requirements.txt") from exc


BENCHMARK_METRICS = ("cer", "normalized_cer", "wer", "normalized_wer")


def _peak_rss_mib(process: subprocess.Popen[str], poll_seconds: float = 0.05) -> tuple[float, int]:
    """Poll process-tree RSS and return peak MiB plus exit code."""
    try:
        root = psutil.Process(process.pid)
    except psutil.Error:
        root = None
    peak = 0
    while process.poll() is None:
        total = 0
        if root is not None:
            try:
                total += root.memory_info().rss
                for child in root.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
        peak = max(peak, total)
        time.sleep(poll_seconds)
    if root is not None:
        try:
            peak = max(peak, root.memory_info().rss)
        except psutil.Error:
            pass
    return peak / (1024 * 1024), int(process.returncode or 0)


def _benchmark_oracle(page: dict[str, Any]) -> dict[str, Any] | None:
    """Describe best candidate by reference metrics; NEVER controls runtime selection."""
    candidates = page.get("candidates", [])
    baseline = next((c for c in candidates if c.get("name") == "original"), None)
    available = [c for c in candidates if isinstance(c.get("metrics"), dict)]
    if not any(any(isinstance(c["metrics"].get(metric), (int, float)) for c in available)
               for metric in BENCHMARK_METRICS):
        return None

    result: dict[str, Any] = {
        "runtime_selected_candidate": page.get("selected_variant", "original"),
        "ground_truth_used_for_runtime_selection": False,
        "best_candidates_by_metric": {},
    }
    for metric in BENCHMARK_METRICS:
        scored = [
            c for c in available
            if isinstance(c.get("metrics", {}).get(metric), (int, float))
        ]
        if not scored:
            continue
        best = min(scored, key=lambda c: float(c["metrics"][metric]))
        baseline_value = (
            float(baseline["metrics"][metric])
            if baseline and isinstance(baseline.get("metrics", {}).get(metric), (int, float))
            else None
        )
        best_value = float(best["metrics"][metric])
        delta = (baseline_value - best_value) if baseline_value is not None else None
        relative = (
            (delta / baseline_value * 100.0)
            if delta is not None and baseline_value not in (None, 0.0)
            else None
        )
        result["best_candidates_by_metric"][metric] = {
            "candidate": best.get("name"),
            "value": round(best_value, 6),
            "baseline_value": None if baseline_value is None else round(baseline_value, 6),
            "absolute_improvement_vs_baseline": None if delta is None else round(delta, 6),
            "relative_error_reduction_pct": None if relative is None else round(relative, 2),
            "runtime_selection_matches_benchmark_oracle": best.get("name") == result["runtime_selected_candidate"],
        }
    return result


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = [
        "# Arabic Doc AI v0.3.1 — Decision Engine 2.0 benchmark",
        "",
        f"Input: `{report['input']}`",
        f"Elapsed: {report['elapsed_seconds']:.2f} s",
        f"Peak process-tree RSS: {report['peak_rss_mib']:.1f} MiB",
        f"Ground truth supplied: `{report['ground_truth_supplied']}`",
        "",
        "| Page | Candidate | Selected at runtime | Heuristic score | Mean conf. | Lines | Strict CER | Norm. CER | Strict WER | Norm. WER |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for page in report["pages"]:
        for candidate in page["candidates"]:
            metrics = candidate.get("metrics", {})
            def fmt(key: str) -> str:
                value = metrics.get(key)
                return "—" if not isinstance(value, (int, float)) else f"{value:.4f}"
            confidence = metrics.get("mean_confidence")
            score = candidate.get("adjusted_score")
            rows.append(
                "| {page} | {name} | {selected} | {score} | {confidence} | {lines} | "
                "{cer} | {ncer} | {wer} | {nwer} |".format(
                    page=page["page_number"],
                    name=candidate.get("name", "unknown"),
                    selected="yes" if candidate.get("selected") else "no",
                    score="—" if not isinstance(score, (int, float)) else f"{score:.4f}",
                    confidence="—" if not isinstance(confidence, (int, float)) else f"{confidence:.3f}",
                    lines=metrics.get("line_count", "—"),
                    cer=fmt("cer"), ncer=fmt("normalized_cer"),
                    wer=fmt("wer"), nwer=fmt("normalized_wer"),
                )
            )

        oracle = page.get("benchmark_oracle")
        if oracle:
            rows.extend(["", f"### Page {page['page_number']} — benchmark-only oracle", ""])
            rows.append(
                f"Runtime selected: **{oracle['runtime_selected_candidate']}**. "
                "The following candidates are best only against the supplied reference; "
                "this information did not influence runtime selection."
            )
            rows.append("")
            rows.append("| Metric | Best candidate | Best value | Error reduction vs original | Runtime matched? |")
            rows.append("|---|---|---:|---:|:---:|")
            for metric, info in oracle["best_candidates_by_metric"].items():
                reduction = info["relative_error_reduction_pct"]
                rows.append(
                    f"| {metric} | {info['candidate']} | {info['value']:.4f} | "
                    f"{'—' if reduction is None else f'{reduction:.1f}%'} | "
                    f"{'yes' if info['runtime_selection_matches_benchmark_oracle'] else 'no'} |"
                )
            rows.append("")

    rows.extend([
        "",
        "## Interpretation and safeguards",
        "",
        "- Strict CER/WER preserve Arabic letter forms, digits, and punctuation; whitespace is normalized.",
        "- Normalized CER/WER additionally fold common alef variants, Arabic/Persian digit forms, diacritics, and tatweel.",
        "- Normalized metrics are a complementary view, not a replacement for strict scoring.",
        "- Benchmark-oracle rankings use the supplied transcription and are evaluation-only; they must never be used to choose a live result.",
        "- Verify the transcription independently. A mistaken reference can make the wrong candidate appear best.",
        "- Candidate heuristic scores and OCR confidence are not measured accuracy.",
        "- Candidates are evaluated sequentially so the OCR model is loaded once.",
        "",
    ])
    (output_dir / "comparison.md").write_text("\n".join(rows), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run adaptive preprocessing and produce Decision Engine 2.0 benchmark reports."
    )
    parser.add_argument("input", help="Input PDF or image path")
    parser.add_argument("--output-dir", default="data/output/adaptive-evaluation")
    parser.add_argument("--ground-truth", help="Optional UTF-8 manual transcription; use form-feed between PDF pages")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max-side", type=int, default=2800)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--selection-margin", type=float, default=0.015)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        print(f"Input file not found: {source}", file=sys.stderr)
        return 2
    if not 1 <= args.threads <= 8:
        print("--threads must be between 1 and 8", file=sys.stderr)
        return 2
    if not 0 <= args.max_candidates <= 5:
        print("--max-candidates must be between 0 and 5", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "ocr-result.json"
    log_path = output_dir / "run.log"

    command = [
        sys.executable, "-m", "arabic_doc_ai.cli", str(source),
        "--dpi", str(args.dpi),
        "--max-side", str(args.max_side),
        "--threads", str(args.threads),
        "--preprocess-mode", "adaptive",
        "--max-preprocessing-candidates", str(args.max_candidates),
        "--selection-margin", str(args.selection_margin),
        "--output", str(result_path),
    ]
    if args.max_pages is not None:
        command.extend(["--max-pages", str(args.max_pages)])
    if args.ground_truth:
        truth_path = Path(args.ground_truth).expanduser().resolve()
        if not truth_path.is_file():
            print(f"Ground-truth file not found: {truth_path}", file=sys.stderr)
            return 2
        command.extend(["--ground-truth", str(truth_path)])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["DISABLE_MODEL_SOURCE_CHECK"] = "True"

    print("Running Decision Engine 2.0: original first, then bounded candidates.", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        peak_rss_mib, return_code = _peak_rss_mib(process)
    elapsed = time.perf_counter() - started

    if return_code != 0 or not result_path.is_file():
        print(f"OCR run failed. See {log_path}", file=sys.stderr)
        if log_path.is_file():
            print(log_path.read_text(encoding="utf-8", errors="replace")[-5000:], file=sys.stderr)
        return return_code or 1

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    pages = []
    for page in payload.get("pages", []):
        preprocessing = page.get("preprocessing", {})
        page_report = {
            "page_number": page.get("page_number"),
            "selected_variant": preprocessing.get("selected_variant", "original"),
            "original_quality": preprocessing.get("original_quality", {}),
            "selection_method": preprocessing.get("selection_method"),
            "candidates": preprocessing.get("candidates", []),
        }
        page_report["benchmark_oracle"] = (
            _benchmark_oracle(page_report) if args.ground_truth else None
        )
        pages.append(page_report)

    report = {
        "input": str(source),
        "device": "cpu",
        "threads": args.threads,
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_mib": round(peak_rss_mib, 1),
        "ground_truth_supplied": bool(args.ground_truth),
        "ground_truth_used_for_runtime_selection": False,
        "ocr_result_file": result_path.name,
        "run_log_file": log_path.name,
        "pages": pages,
    }
    _write_reports(output_dir, report)
    print(f"\nReport: {output_dir / 'comparison.md'}")
    print(f"OCR JSON: {result_path}")
    print(f"Elapsed: {elapsed:.1f}s | Peak RSS: {peak_rss_mib:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
