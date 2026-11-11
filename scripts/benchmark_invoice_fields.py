#!/usr/bin/env python3
"""Evaluate selected OCR output against structured invoice ground-truth fields.

This runner does NOT use labels to choose preprocessing candidates. It runs the
OCR pipeline first, then evaluates the selected result against field labels.
It reports exact normalized matches and near matches separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

def _normalize_arabic(value: str) -> str:
    # Kept local so unit tests for field scoring do not require Paddle/PaddleOCR.
    import unicodedata
    text = unicodedata.normalize("NFKC", value).translate(DIGIT_TRANSLATION)
    text = text.replace("ـ", "")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()

DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"
)
ARABIC_DECIMAL = "٫"
ARABIC_THOUSANDS = "٬"
ARABIC_COMMA = "،"
SKIP_FIELDS = {"language", "date_calendar", "invoice_date"}
NUMERIC_FIELDS = {"qty", "subtotal", "unit_price", "vat_amount", "total", "total_raw"}


def compact_text(value: str) -> str:
    """Conservative comparison form; raw OCR text remains untouched."""
    text = _normalize_arabic(unicodedata.normalize("NFKC", value))
    return "".join(ch for ch in text if ch.isalnum())


def decimal_candidates(text: str) -> list[Decimal]:
    """Parse OCR-like numbers, including Arabic digits and separators."""
    text = unicodedata.normalize("NFKC", text).translate(DIGIT_TRANSLATION)
    text = text.replace(ARABIC_THOUSANDS, "")
    text = text.replace(ARABIC_DECIMAL, ".")
    text = text.replace(ARABIC_COMMA, ".")
    # Keep a dot as a decimal mark. A normal comma is treated similarly because
    # invoice OCR often renders decimal commas as ASCII commas.
    text = text.replace(",", ".")
    values: list[Decimal] = []
    for token in re.findall(r"(?<![\w.])\d+(?:\.\d+)*(?![\w.])", text):
        if token.count(".") > 1:
            # Treat all but the final separator as grouping separators.
            head, tail = token.rsplit(".", 1)
            token = head.replace(".", "") + "." + tail
        try:
            values.append(Decimal(token))
        except InvalidOperation:
            continue
    return values


def expected_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        # Dataset values are numeric scalars or strings such as Arabic-formatted amounts.
        parsed = decimal_candidates(str(value))
        return parsed[0] if parsed else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def best_line_similarity(expected: str, lines: list[str]) -> float:
    target = compact_text(expected)
    if not target:
        return 0.0
    scores = [SequenceMatcher(None, target, compact_text(line)).ratio() for line in lines]
    return max(scores, default=SequenceMatcher(None, target, "").ratio())


def evaluate_fields(ground_truth: dict[str, Any], ocr_text: str, ocr_lines: list[dict[str, Any]]) -> dict[str, Any]:
    raw_lines = [str(line.get("text", "")) for line in ocr_lines if line.get("text")]
    normalized_all = compact_text(ocr_text)
    line_values = raw_lines or [ocr_text]
    numeric_values = decimal_candidates(ocr_text)
    results: list[dict[str, Any]] = []

    for field, expected in ground_truth.items():
        if field in SKIP_FIELDS or expected is None:
            continue

        if field == "currency":
            aliases = {"AED": ["درهم", "درهما", "درهم إماراتي", "aed"]}
            candidates = aliases.get(str(expected).upper(), [str(expected)])
            found = any(compact_text(alias) in normalized_all for alias in candidates)
            results.append({"field": field, "expected": expected,
                            "status": "matched" if found else "not_found",
                            "match_type": "currency_alias", "similarity": 1.0 if found else 0.0})
            continue

        if field in NUMERIC_FIELDS:
            numeric_expected = expected_decimal(expected)
            found = numeric_expected is not None and any(
                value == numeric_expected for value in numeric_values
            )
            results.append({"field": field, "expected": expected,
                            "status": "matched" if found else "not_found",
                            "match_type": "numeric_exact", "similarity": 1.0 if found else 0.0})
            continue

        expected_str = str(expected)
        target = compact_text(expected_str)
        found = bool(target and target in normalized_all)
        similarity = 1.0 if found else best_line_similarity(expected_str, line_values)
        status = "matched" if found else ("near_match" if similarity >= 0.82 else "not_found")
        results.append({"field": field, "expected": expected,
                        "status": status, "match_type": "normalized_text",
                        "similarity": round(similarity, 4)})

    total = len(results)
    matched = sum(item["status"] == "matched" for item in results)
    near = sum(item["status"] == "near_match" for item in results)
    missing = total - matched - near
    return {
        "scored_fields": total,
        "matched_fields": matched,
        "near_match_fields": near,
        "not_found_fields": missing,
        "exact_field_match_rate": round(matched / total, 4) if total else None,
        "field_results": results,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def select_records(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    labels = read_jsonl(dataset_root / "annotations" / "ground_truth.jsonl")
    by_id = {row["document_id"]: row for row in labels}
    if split == "all":
        return labels
    refs = read_jsonl(dataset_root / "splits" / f"{split}.jsonl")
    missing = [ref["document_id"] for ref in refs if ref["document_id"] not in by_id]
    if missing:
        raise ValueError(f"Split references missing labels: {missing[:3]}")
    return [by_id[ref["document_id"]] for ref in refs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark selected OCR fields against invoice ground truth.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/pilot/hv09_invoices_200"))
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="train")
    parser.add_argument("--limit", type=int, default=1, help="Maximum documents; use 0 for all records in the split.")
    parser.add_argument("--mode", choices=("baseline", "adaptive"), default="baseline")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("data/output/invoice-benchmark"))
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.threads < 1 or args.threads > 8:
        parser.error("--threads must be between 1 and 8")

    dataset_root = args.dataset.expanduser().resolve()
    records = select_records(dataset_root, args.split)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("No records selected. Check dataset path and split files.")

    try:
        from arabic_doc_ai.config import OCRConfig
        from arabic_doc_ai.pipeline import ArabicDocumentOCR
    except ImportError as exc:
        raise SystemExit(
            "Could not import arabic_doc_ai. Activate the project's .venv and run "
            "`python -m pip install -e .` first."
        ) from exc

    config = OCRConfig(
        threads=args.threads,
        preprocess_mode=args.mode,
        preprocess_enabled=(args.mode == "adaptive"),
        max_preprocessing_candidates=args.max_candidates,
        max_pages=1,
    )
    ocr = ArabicDocumentOCR(config)  # one lazily loaded model, reused sequentially
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    start_all = time.perf_counter()

    for index, record in enumerate(records, 1):
        image_path = (dataset_root / record["image_path"]).resolve()
        if not image_path.is_file():
            errors.append({"document_id": record["document_id"], "error": f"Missing image: {image_path}"})
            continue
        start = time.perf_counter()
        try:
            ocr_result = ocr.extract(image_path)
            page = ocr_result["pages"][0] if ocr_result.get("pages") else {"text": "", "lines": []}
            field_metrics = evaluate_fields(
                record.get("ground_truth_fields", {}),
                page.get("text", ""), page.get("lines", []),
            )
            row = {
                "document_id": record["document_id"],
                "language": record.get("language"),
                "image_path": record["image_path"],
                "selected_variant": page.get("preprocessing", {}).get("selected_variant", "original"),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                **field_metrics,
                "ocr_text": page.get("text", ""),
            }
            results.append(row)
            print(
                f"[{index}/{len(records)}] {record['document_id']}: "
                f"{field_metrics['matched_fields']}/{field_metrics['scored_fields']} fields matched "
                f"(variant={row['selected_variant']}, {row['elapsed_seconds']}s)"
            )
        except Exception as exc:  # continue pilot; record per-document failure
            errors.append({"document_id": record["document_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(records)}] ERROR {record['document_id']}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - start_all
    total_fields = sum(row["scored_fields"] for row in results)
    total_matches = sum(row["matched_fields"] for row in results)
    total_near = sum(row["near_match_fields"] for row in results)
    total_missing = sum(row["not_found_fields"] for row in results)
    summary = {
        "benchmark_version": "invoice-field-pilot-0.1",
        "dataset": str(dataset_root),
        "split": args.split,
        "mode": args.mode,
        "documents_requested": len(records),
        "documents_completed": len(results),
        "documents_failed": len(errors),
        "elapsed_seconds": round(elapsed, 3),
        "scored_fields": total_fields,
        "matched_fields": total_matches,
        "near_match_fields": total_near,
        "not_found_fields": total_missing,
        "micro_exact_field_match_rate": round(total_matches / total_fields, 4) if total_fields else None,
        "errors": errors,
        "ground_truth_policy": "Labels are used only after OCR for evaluation; they never affect preprocessing selection.",
        "caveat": "This is field-level evaluation, not full-page CER/WER. Fuzzy near matches are not counted as exact matches.",
    }

    (output_dir / "results.json").write_text(json.dumps({"summary": summary, "documents": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "document_id", "language", "selected_variant", "elapsed_seconds",
            "scored_fields", "matched_fields", "near_match_fields", "not_found_fields",
            "exact_field_match_rate",
        ])
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    lines = [
        "# Arabic Doc AI — Invoice Field Benchmark",
        "",
        f"- Split: `{args.split}`",
        f"- Mode: `{args.mode}`",
        f"- Documents: {len(results)} completed / {len(records)} requested",
        f"- Elapsed: {elapsed:.2f}s",
        f"- Exact field matches: {total_matches}/{total_fields} ({summary['micro_exact_field_match_rate']})",
        f"- Near matches (not counted as exact): {total_near}",
        f"- Not found: {total_missing}",
        "",
        "| Document | Language | Variant | Exact | Near | Missing | Exact rate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['document_id']} | {row.get('language')} | {row['selected_variant']} | "
            f"{row['matched_fields']}/{row['scored_fields']} | {row['near_match_fields']} | "
            f"{row['not_found_fields']} | {row['exact_field_match_rate']} |"
        )
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{item['document_id']}`: {item['error']}" for item in errors)
    lines.extend([
        "", "## Interpretation", "",
        "- This benchmark scores structured fields only; it does not calculate full-page CER/WER.",
        "- Ground truth is applied after OCR and never influences runtime candidate selection.",
        "- Near matches are reported separately and are not counted as correct.",
        "- Review field-level examples before treating aggregate rates as model accuracy.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nBenchmark complete.")
    print(f"Report: {output_dir / 'report.md'}")
    print(f"JSON:   {output_dir / 'results.json'}")
    print(f"CSV:    {output_dir / 'summary.csv'}")
    print(f"Exact field match: {total_matches}/{total_fields}")
    print(f"Errors: {len(errors)}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
