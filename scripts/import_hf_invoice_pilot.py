#!/usr/bin/env python3
"""Download a small, reproducible Arabic/bilingual invoice pilot dataset.

Source: HV09/synthetic-bilingual-invoices-200 (CC-BY-4.0).
The source contains fictional invoices and 17 structured ground-truth fields.
This script intentionally imports only a small sample; it does not run OCR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

DATASET_ID = "HV09/synthetic-bilingual-invoices-200"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
LICENSE = "CC-BY-4.0"
DEFAULT_LANGUAGES = ("ar", "bilingual")


def jsonable(value: Any) -> Any:
    """Convert dataset values to JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    # NumPy scalar support without importing NumPy explicitly.
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)


def save_image(image_value: Any, output_path: Path) -> str:
    """Save a Hugging Face Image feature to PNG and return a SHA-256 digest."""
    from PIL import Image
    from io import BytesIO

    image = image_value
    if isinstance(image, dict):
        if image.get("bytes"):
            image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path"):
            image = Image.open(image["path"])
        else:
            raise ValueError("Image feature is a dict without bytes/path")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image value: {type(image)!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image.save(output_path, format="PNG", optimize=True)
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stratified_splits(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic per-language 70/15/15 splits."""
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["language"], []).append(record)

    result = {"train": [], "validation": [], "test": []}
    for language in sorted(grouped):
        group = list(grouped[language])
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * 0.70))
        n_validation = max(1, int(n * 0.15)) if n >= 7 else 0
        # Keep at least one item in test whenever a language has >= 3 rows.
        if n >= 3:
            n_validation = min(n_validation, n - n_train - 1)
        train = group[:n_train]
        validation = group[n_train:n_train + n_validation]
        test = group[n_train + n_validation:]
        result["train"].extend(train)
        result["validation"].extend(validation)
        result["test"].extend(test)

    for split in result:
        rng.shuffle(result[split])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a small Arabic invoice OCR benchmark pilot.")
    parser.add_argument("--per-language", type=int, default=20,
                        help="Number of rows per language (default: 20; source has 50 Arabic and 50 bilingual rows).")
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES),
                        choices=["ar", "bilingual", "en", "en-au"],
                        help="Language styles to include (default: ar bilingual).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("datasets/pilot/hv09_invoices_200"))
    args = parser.parse_args()

    if args.per_language < 1:
        parser.error("--per-language must be >= 1")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency. Install it with: python -m pip install datasets", file=sys.stderr)
        return 2

    print(f"Loading public dataset: {DATASET_ID}")
    print("The source has 200 synthetic invoices and is about 7.37 MB in total.")
    print("Only the selected rows will be saved to the project output folder.")
    try:
        dataset = load_dataset(DATASET_ID, split="train")
    except Exception as exc:
        print(f"Could not load the dataset: {exc}", file=sys.stderr)
        print("Check your internet connection and try again.", file=sys.stderr)
        return 1

    buckets: dict[str, list[dict[str, Any]]] = {lang: [] for lang in args.languages}
    for row in dataset:
        lang = str(row.get("language", "")).strip().lower()
        if lang in buckets and len(buckets[lang]) < args.per_language:
            buckets[lang].append(row)
        if all(len(items) >= args.per_language for items in buckets.values()):
            break

    for lang, rows in buckets.items():
        if not rows:
            print(f"Warning: no rows found for language style {lang!r}", file=sys.stderr)

    output = args.output.resolve()
    images_root = output / "images"
    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for language in args.languages:
        for row in buckets[language]:
            image_value = row.get("image")
            if image_value is None:
                print(f"Skipping row without image in language {language}", file=sys.stderr)
                continue

            # Hash the bytes after saving to obtain a stable, content-derived ID.
            temp_name = f"sample_{len(records) + 1:04d}.png"
            temp_path = images_root / language / temp_name
            digest = save_image(image_value, temp_path)
            doc_id = f"{language}_{digest[:12]}"
            final_path = images_root / language / f"{doc_id}.png"
            if final_path.exists():
                temp_path.unlink(missing_ok=True)
            else:
                temp_path.replace(final_path)

            ground_truth = {k: jsonable(v) for k, v in row.items() if k != "image"}
            record = {
                "document_id": doc_id,
                "source_dataset": DATASET_ID,
                "source_url": DATASET_URL,
                "source_license": LICENSE,
                "language": language,
                "image_path": str(final_path.relative_to(output)),
                "ground_truth_fields": ground_truth,
                "ground_truth_type": "structured_fields_not_full_page_transcription",
            }
            records.append(record)
            manifest_rows.append({
                "document_id": doc_id,
                "language": language,
                "image_path": record["image_path"],
                "invoice_number": ground_truth.get("invoice_number"),
                "currency": ground_truth.get("currency"),
                "source_dataset": DATASET_ID,
                "license": LICENSE,
            })

    if not records:
        print("No records were imported. Nothing was written.", file=sys.stderr)
        return 1

    write_jsonl(output / "annotations" / "ground_truth.jsonl", records)
    splits = stratified_splits(records, args.seed)
    for split_name, split_records in splits.items():
        # Store document IDs only; canonical labels remain in ground_truth.jsonl.
        refs = [{"document_id": r["document_id"], "image_path": r["image_path"]} for r in split_records]
        write_jsonl(output / "splits" / f"{split_name}.jsonl", refs)

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = f"""# Arabic Doc AI — Dataset Pilot\n\n- Source: [{DATASET_ID}]({DATASET_URL})\n- License: {LICENSE} (retain attribution when redistributing or publishing derived datasets).\n- Contents: {len(records)} downloaded invoice images from styles: {', '.join(args.languages)}.\n- Selection: up to {args.per_language} rows per style; seed={args.seed}.\n- Source data: fictional/synthetic invoices, not real business records.\n- Ground truth: structured invoice fields (17 fields in the source), **not a complete line-by-line page transcription**. Use field-level accuracy for this pilot; do not report full-page CER/WER from these fields.\n- Splits: deterministic 70/15/15 per language style. Keep test split untouched while developing heuristics.\n\n## Files\n\n- `images/`: downloaded PNGs.\n- `annotations/ground_truth.jsonl`: structured labels and provenance.\n- `manifest.csv`: compact index.\n- `splits/`: train/validation/test document IDs.\n\n## Attribution\n\nDataset: Synthetic Bilingual Arabic/English Invoices — 200 documents with per-field ground truth, published by HV09, CC-BY-4.0.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    print("\nImport complete.")
    print(f"Output: {output}")
    print(f"Total images: {len(records)}")
    for language in args.languages:
        print(f"  {language}: {sum(r['language'] == language for r in records)}")
    print(f"Splits: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")
    print("Ground truth is structured field-level data, not a full-page transcript.")
    print("No OCR was run and no Paddle model was loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
