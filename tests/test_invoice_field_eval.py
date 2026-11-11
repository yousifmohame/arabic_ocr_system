import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_invoice_fields.py"
SPEC = importlib.util.spec_from_file_location("benchmark_invoice_fields", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InvoiceFieldEvaluationTests(unittest.TestCase):
    def test_arabic_digits_and_decimal_amount(self):
        result = MODULE.evaluate_fields(
            {"qty": 308, "total": 44548.35},
            "الكمية ٣٠٨\nالإجمالي ٤٤٥٤٨،٣٥",
            [{"text": "الكمية ٣٠٨"}, {"text": "الإجمالي ٤٤٥٤٨،٣٥"}],
        )
        self.assertEqual(result["matched_fields"], 2)

    def test_text_matching_is_normalized(self):
        result = MODULE.evaluate_fields(
            {"description": "زبدة غير مملحة ٥ كجم"},
            "زبدة غير مملحة 5 كجم",
            [{"text": "زبدة غير مملحة 5 كجم"}],
        )
        self.assertEqual(result["matched_fields"], 1)

    def test_missing_value_is_not_guessed(self):
        result = MODULE.evaluate_fields(
            {"supplier_trn": "100647382910003"},
            "بستان النخيل للتغليف",
            [{"text": "بستان النخيل للتغليف"}],
        )
        self.assertEqual(result["not_found_fields"], 1)
        self.assertEqual(result["matched_fields"], 0)

    def test_metadata_and_null_fields_are_skipped(self):
        result = MODULE.evaluate_fields(
            {"language": "ar", "date_calendar": "gregorian", "supplier": None, "qty": 308},
            "٣٠٨", [{"text": "٣٠٨"}],
        )
        self.assertEqual(result["scored_fields"], 1)


if __name__ == "__main__":
    unittest.main()
