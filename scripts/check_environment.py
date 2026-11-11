from __future__ import annotations

import importlib.util
import platform
import sys


def main() -> None:
    print("Arabic Document OCR - environment report")
    print("=" * 44)
    print(f"OS: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    for module in ("paddle", "paddleocr", "fitz", "PIL", "cv2", "numpy"):
        available = importlib.util.find_spec(module) is not None
        print(f"{module}: {'installed' if available else 'not installed'}")

    if platform.system() == "Darwin":
        print("\nMac note: verify that your macOS version and CPU architecture are supported by the selected PaddlePaddle wheel.")
    print("\nNo model was loaded and no network request was made by this script.")


if __name__ == "__main__":
    main()
