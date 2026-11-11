from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_doc_ai.config import (  # noqa: E402
    ARABIC_RECOGNITION_MODEL_DIR,
    DETECTION_MODEL_DIR,
    OCRConfig,
)
from arabic_doc_ai.engine import PaddleArabicOCREngine  # noqa: E402

MODEL_FILES = [
    "inference.yml",
    "inference.json",
    "inference.pdiparams",
    "config.json",
]


def download_model(repo_id: str, destination: Path) -> None:
    """Download an exported Paddle inference model into the project directory."""
    required = (destination / "inference.yml", destination / "inference.pdiparams")
    if all(path.is_file() for path in required):
        print(f"✓ النموذج موجود محليًا: {destination}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install project requirements first."
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    print(f"تنزيل {repo_id} إلى {destination}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        allow_patterns=MODEL_FILES,
    )

    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Model repository {repo_id} did not provide required files: {missing}"
        )


def main() -> int:
    print("تجهيز النماذج المحلية. يتطلب هذا التنزيل الأول اتصالًا بالإنترنت.")
    try:
        download_model("PaddlePaddle/PP-OCRv5_mobile_det", DETECTION_MODEL_DIR)
        download_model(
            "PaddlePaddle/arabic_PP-OCRv5_mobile_rec",
            ARABIC_RECOGNITION_MODEL_DIR,
        )

        print()
        print("اختبار تحميل النماذج على CPU...")
        engine = PaddleArabicOCREngine(OCRConfig())
        engine._load()
    except Exception as exc:
        print(f"فشل تجهيز النماذج: {exc}", file=sys.stderr)
        return 1

    print("✓ تم تحميل وتهيئة النماذج المحلية.")
    print("اختبر وثيقة فعلية قبل فصل الإنترنت.")
    print(f"كاشف النصوص: {DETECTION_MODEL_DIR}")
    print(f"التعرف العربي: {ARABIC_RECOGNITION_MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
