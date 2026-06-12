from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from image_aug_cli.config import OutputConfig


def iter_images(input_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def read_rgb_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("OpenCV could not decode the image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb_image(path: Path, image_rgb: np.ndarray, output: OutputConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    extension = path.suffix.lower()
    params: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), output.jpeg_quality]
    elif extension == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), output.png_compression]

    success, encoded = cv2.imencode(extension, image_bgr, params)
    if not success:
        raise ValueError(f"OpenCV could not encode output image as {extension}.")

    encoded.tofile(str(path))


def build_output_path(
    src_path: Path,
    input_dir: Path,
    output_dir: Path,
    index: int,
    output: OutputConfig,
) -> Path:
    relative = src_path.relative_to(input_dir)
    parent = output_dir / relative.parent if output.preserve_tree else output_dir
    extension = _normalize_output_extension(output.format) or src_path.suffix.lower()
    suffix = output.suffix.format(index=index)
    return parent / f"{src_path.stem}{suffix}{extension}"


def _normalize_output_extension(format_name: str | None) -> str | None:
    if not format_name:
        return None
    normalized = format_name.lower().strip()
    if normalized == "jpg":
        normalized = "jpeg"
    return f".{normalized.lstrip('.')}"
