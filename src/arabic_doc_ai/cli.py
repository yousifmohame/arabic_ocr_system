from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import OCRConfig
from .pipeline import ArabicDocumentOCR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arabic-doc-ocr",
        description="Local CPU-only Arabic OCR for images and PDF files.",
    )
    parser.add_argument("input", help="Path to a PDF or image file")
    parser.add_argument("-o", "--output", help="Output JSON path (default: <input>.ocr.json)")
    parser.add_argument("--ground-truth", help="Optional manually transcribed UTF-8 text for CER/WER evaluation (does not affect selection)")
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI (default: 200)")
    parser.add_argument("--max-side", type=int, default=2800, help="Resize longest image side to this value (default: 2800)")
    parser.add_argument("--max-pages", type=int, help="Process only the first N PDF pages")
    parser.add_argument("--threads", type=int, default=2, help="CPU inference threads (default: 2)")
    parser.add_argument(
        "--preprocess-mode",
        choices=("adaptive", "baseline"),
        default="adaptive",
        help="adaptive compares selected transforms with the original; baseline runs original only",
    )
    parser.add_argument(
        "--max-preprocessing-candidates",
        type=int,
        default=3,
        help="Maximum additional OCR passes per page (default: 3)",
    )
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=0.015,
        help="Required heuristic score improvement over original (default: 0.015)",
    )
    parser.add_argument("--no-preprocess", action="store_true", help="Alias for --preprocess-mode baseline")
    parser.add_argument("--no-perspective-correction", action="store_true", help="Do not detect/rectify photographed document boundaries")
    parser.add_argument("--no-deskew", action="store_true", help="Disable the Hough-based deskew candidate")
    parser.add_argument("--no-enhance", action="store_true", help="Disable CLAHE and mild sharpening")
    parser.add_argument("--no-upscale", action="store_true", help="Disable upscaling of small images")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser() if args.output else source.with_suffix(source.suffix + ".ocr.json")

    try:
        config = OCRConfig(
            dpi=args.dpi,
            max_side=args.max_side,
            max_pages=args.max_pages,
            threads=args.threads,
            preprocess_enabled=not args.no_preprocess,
            preprocess_mode="baseline" if args.no_preprocess else args.preprocess_mode,
            max_preprocessing_candidates=args.max_preprocessing_candidates,
            preprocessing_selection_margin=args.selection_margin,
            correct_perspective=not args.no_perspective_correction,
            correct_skew=not args.no_deskew,
            enhance_images=not args.no_enhance,
            upscale_small_images=not args.no_upscale,
        )
        ground_truth_text = None
        if args.ground_truth:
            ground_truth_path = Path(args.ground_truth).expanduser()
            if not ground_truth_path.is_file():
                raise FileNotFoundError(f"Ground-truth file not found: {ground_truth_path}")
            ground_truth_text = ground_truth_path.read_text(encoding="utf-8")
        result: dict[str, Any] = ArabicDocumentOCR(config).extract(
            source, ground_truth_text=ground_truth_text
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        text_path = output_path.with_suffix(".txt")
        text_content = "\n\n".join(
            f"--- الصفحة {page['page_number']} ---\n{page['text']}"
            for page in result["pages"]
        )
        text_path.write_text(text_content, encoding="utf-8")

        print(f"تمت معالجة {result['page_count']} صفحة.")
        for page in result["pages"]:
            decision = page.get("preprocessing", {})
            print(
                f"الصفحة {page['page_number']}: اختير "
                f"{decision.get('selected_variant', 'original')} "
                f"(score={decision.get('selection_score', 'n/a')})"
            )
            for candidate in decision.get("candidates", []):
                marker = "*" if candidate.get("selected") else " "
                metrics = candidate.get("metrics", {})
                cer_value = metrics.get("cer")
                cer_display = "n/a" if cer_value is None else f"{cer_value:.4f}"
                print(
                    f"  {marker} {candidate.get('name')}: "
                    f"score={candidate.get('adjusted_score')}, "
                    f"CER={cer_display}"
                )
        print(f"JSON: {output_path.resolve()}")
        print(f"النص: {text_path.resolve()}")
        return 0
    except Exception as exc:
        print(f"خطأ: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
