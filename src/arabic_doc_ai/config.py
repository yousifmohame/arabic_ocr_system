from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OCRConfig:
    """Runtime settings for low-memory, CPU-only OCR."""

    language: str = "ar"
    ocr_version: str = "PP-OCRv5"
    device: str = "cpu"
    dpi: int = 200
    max_side: int = 2800
    max_pages: int | None = None
    threads: int = 2
    use_document_orientation: bool = False
    use_document_unwarping: bool = False
    use_textline_orientation: bool = False
    recognition_batch_size: int = 1
    preprocess_enabled: bool = True
    preprocess_mode: str = "adaptive"
    max_preprocessing_candidates: int = 3
    preprocessing_selection_margin: float = 0.015
    correct_perspective: bool = True
    correct_skew: bool = True
    enhance_images: bool = True
    upscale_small_images: bool = True
    perspective_min_boundary_score: float = 0.22
    perspective_min_severity: float = 0.025
    perspective_max_area_ratio: float = 0.97
    deskew_trigger_degrees: float = 1.5
    deskew_min_line_support: int = 2
    blur_variance_threshold: float = 85.0
    enhance_dynamic_range_threshold: float = 105.0
    enhance_contrast_threshold: float = 46.0
    dark_brightness_threshold: float = 72.0
    bright_brightness_threshold: float = 218.0
    illumination_variation_threshold: float = 36.0
    upscale_trigger_long_side: int = 900
    preprocess_upscale_threshold: int = 1200
    preprocess_target_long_side: int = 1200
    preprocess_max_upscale: float = 2.0
    preprocess_max_side: int = 2400

    def validate(self) -> None:
        if self.device != "cpu":
            raise ValueError("This build is intentionally CPU-only.")
        if self.dpi < 72 or self.dpi > 400:
            raise ValueError("dpi must be between 72 and 400.")
        if self.max_side < 800:
            raise ValueError("max_side must be at least 800 pixels.")
        if self.threads < 1 or self.threads > 8:
            raise ValueError("threads must be between 1 and 8.")
        if self.max_pages is not None and self.max_pages < 1:
            raise ValueError("max_pages must be positive when provided.")
        if self.preprocess_mode not in {"adaptive", "baseline"}:
            raise ValueError("preprocess_mode must be 'adaptive' or 'baseline'.")
        if self.max_preprocessing_candidates < 0 or self.max_preprocessing_candidates > 5:
            raise ValueError("max_preprocessing_candidates must be between 0 and 5.")
        if not 0.0 <= self.preprocessing_selection_margin <= 0.25:
            raise ValueError("preprocessing_selection_margin must be between 0 and 0.25.")
        if self.deskew_trigger_degrees < 0.5 or self.deskew_trigger_degrees > 15:
            raise ValueError("deskew_trigger_degrees must be between 0.5 and 15.")
        if self.deskew_min_line_support < 1 or self.deskew_min_line_support > 20:
            raise ValueError("deskew_min_line_support must be between 1 and 20.")
        if self.blur_variance_threshold < 0:
            raise ValueError("blur_variance_threshold cannot be negative.")
        if not 0.0 <= self.perspective_min_severity <= 1.0:
            raise ValueError("perspective_min_severity must be between 0 and 1.")
        if not 0.0 < self.perspective_max_area_ratio <= 1.0:
            raise ValueError("perspective_max_area_ratio must be between 0 and 1.")
        if self.upscale_trigger_long_side < 400:
            raise ValueError("upscale_trigger_long_side must be at least 400.")
        if self.preprocess_upscale_threshold < 400:
            raise ValueError("preprocess_upscale_threshold must be at least 400.")
        if self.preprocess_target_long_side < 400:
            raise ValueError("preprocess_target_long_side must be at least 400.")
        if self.preprocess_max_upscale < 1.0 or self.preprocess_max_upscale > 3.0:
            raise ValueError("preprocess_max_upscale must be between 1.0 and 3.0.")
        if self.preprocess_max_side < 800:
            raise ValueError("preprocess_max_side must be at least 800.")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DETECTION_MODEL_DIR = MODELS_DIR / "PP-OCRv5_mobile_det"
ARABIC_RECOGNITION_MODEL_DIR = MODELS_DIR / "arabic_PP-OCRv5_mobile_rec"
