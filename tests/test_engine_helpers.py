import unittest

from arabic_doc_ai.engine import sort_reading_order


class ReadingOrderTests(unittest.TestCase):
    def test_top_to_bottom_then_right_to_left(self):
        lines = [
            {"text": "يسار", "bbox": [10, 10, 70, 30]},
            {"text": "يمين", "bbox": [120, 11, 180, 31]},
            {"text": "سطر ثان", "bbox": [20, 60, 100, 80]},
        ]
        ordered = sort_reading_order(lines)
        self.assertEqual([item["text"] for item in ordered], ["يمين", "يسار", "سطر ثان"])


if __name__ == "__main__":
    unittest.main()
