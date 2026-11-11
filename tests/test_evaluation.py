import unittest

from arabic_doc_ai.evaluation import (
    cer, edit_distance, wer, normalized_cer, normalized_wer,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_cer_ignores_whitespace_only(self):
        self.assertEqual(cer("مرحبا بالعالم", "مرحبا  بالعالم"), 0.0)
        self.assertGreater(cer("مبنى", "مبني"), 0.0)

    def test_wer(self):
        self.assertEqual(wer("one two three", "one two three"), 0.0)
        self.assertGreater(wer("one two", "one three"), 0.0)

    def test_normalized_metrics_fold_arabic_digit_forms(self):
        self.assertEqual(normalized_cer("١٢٣٤٥", "12345"), 0.0)
        self.assertEqual(normalized_wer("رقم ١٢٣٤٥", "رقم 12345"), 0.0)
        self.assertGreater(cer("١٢٣٤٥", "12345"), 0.0)

    def test_normalized_metrics_fold_tatweel_diacritics_and_alef_variants(self):
        self.assertEqual(normalized_cer("إِدارة", "ادارة"), 0.0)
        self.assertEqual(normalized_wer("إِدارة", "ادارة"), 0.0)

    def test_normalized_metrics_preserve_maqsurah_vs_yaa(self):
        self.assertGreater(normalized_cer("مبنى", "مبني"), 0.0)


if __name__ == "__main__":
    unittest.main()
