import unittest

from arabic_doc_ai.normalization import normalize_arabic_for_matching, normalize_digits


class NormalizationTests(unittest.TestCase):
    def test_digits(self):
        self.assertEqual(normalize_digits("١٢٣٤٥٦789۰۱۲"), "123456789012")

    def test_arabic_matching_copy(self):
        self.assertEqual(
            normalize_arabic_for_matching("أَحْمَـد   ى"),
            "احمد ي",
        )

    def test_original_is_not_mutated(self):
        original = "أحمد ١٢٣"
        normalized = normalize_arabic_for_matching(original)
        self.assertEqual(original, "أحمد ١٢٣")
        self.assertEqual(normalized, "احمد 123")


if __name__ == "__main__":
    unittest.main()
