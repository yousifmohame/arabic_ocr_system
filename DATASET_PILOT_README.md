# Arabic Doc AI — first dataset pilot

Start with `HV09/synthetic-bilingual-invoices-200`: 200 fictional invoice images, 17 structured ground-truth fields per invoice, four language styles, CC-BY-4.0, total repository size about 7.37 MB. For the first pilot, import 20 Arabic-only and 20 bilingual invoices (40 images total).

The labels are field-level structured labels, not complete page transcriptions. Use them to evaluate field recognition/extraction and compare preprocessing variants on exact field values. Do not calculate full-page CER/WER from concatenated fields.

The dataset card says each style has 50 documents; styles include Arabic-only, bilingual Arabic/English, English, and English-Australian. It is synthetic and contains fictional values.

Official dataset: https://huggingface.co/datasets/HV09/synthetic-bilingual-invoices-200
License shown on the dataset page: CC-BY-4.0. Preserve attribution and review the license terms before redistributing derivative datasets.

## Install and run

Use a separate virtual environment for dataset import so the OCR environment (`.venv`) and PaddleOCR dependencies are not changed:

```bash
cd ~/Desktop/arabic-doc-ai
python3 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install --upgrade pip
python -m pip install datasets pillow
python scripts/import_hf_invoice_pilot.py --per-language 20 --languages ar bilingual
```

Output defaults to `datasets/pilot/hv09_invoices_200/`.

Then leave the data-only environment and return to the OCR environment:

```bash
deactivate
source .venv/bin/activate
```

The downloaded files remain in the project directory. No OCR inference runs during import.
