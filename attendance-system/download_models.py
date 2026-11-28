"""Download the two published OpenCV Zoo ONNX models for offline use."""

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent / "models"
BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
FILES = {
    "face_detection_yunet_2023mar.onnx": "face_detection_yunet",
    "face_recognition_sface_2021dec.onnx": "face_recognition_sface",
}


def main():
    ROOT.mkdir(exist_ok=True)
    for name, folder in FILES.items():
        target = ROOT / name
        if target.exists() and target.stat().st_size > 100_000:
            print(f"Already downloaded: {name}")
            continue
        print(f"Downloading {name}...")
        temp = ROOT / (name + ".part")
        try:
            with urlopen(f"{BASE}/{folder}/{name}", timeout=60) as source, temp.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            if temp.stat().st_size < 100_000:
                raise RuntimeError(f"Invalid model download: {name}")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    print("Models ready. They are stored only on this computer.")


if __name__ == "__main__":
    try:
        main()
    except (URLError, RuntimeError) as exc:
        raise SystemExit(f"Model download failed: {exc}\nSee README.md for manual download links.") from exc
