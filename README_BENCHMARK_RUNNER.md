# Arabic Doc AI — Invoice Field Benchmark Runner (pilot)

This patch adds `scripts/benchmark_invoice_fields.py` and unit tests for comparing OCR text with the dataset's **structured invoice fields**.

It does not train a model and does not use ground truth to choose runtime preprocessing. Labels are read only after OCR has completed.

## Install

Activate the project's OCR environment, then from the project root:

```bash
source .venv/bin/activate
python -m pip install -e .
```

## Run one development example

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export DISABLE_MODEL_SOURCE_CHECK=True

python scripts/benchmark_invoice_fields.py \
  --split train \
  --limit 1 \
  --mode baseline \
  --threads 1 \
  --output-dir data/output/invoice-benchmark-baseline-one
```

## Run the full training split

```bash
python scripts/benchmark_invoice_fields.py \
  --split train --limit 0 --mode baseline --threads 1 \
  --output-dir data/output/invoice-benchmark-train-baseline
```

After baseline is measured, run adaptive mode into a different output folder:

```bash
python scripts/benchmark_invoice_fields.py \
  --split train --limit 0 --mode adaptive --threads 1 --max-candidates 3 \
  --output-dir data/output/invoice-benchmark-train-adaptive
```

Keep `validation` and especially `test` untouched until the ranking rules/model are fixed.

## Outputs

- `report.md`: human-readable summary
- `results.json`: document-level field decisions and OCR text
- `summary.csv`: compact per-document results

The script reports exact normalized matches, near matches, and missing fields separately. Numeric matching normalizes Arabic digits and decimal separators. `supplier`, `language`, `date_calendar`, and ISO `invoice_date` metadata are skipped; `invoice_date_raw` is evaluated as the printed date text. Currency `AED` can match printed `درهم` wording. Fuzzy near matches are never counted as exact.

This is a pilot evaluator, not a production field extractor. In particular, substring matching can produce false positives for short values. Review `results.json` and refine field-specific matching before publishing aggregate accuracy.
