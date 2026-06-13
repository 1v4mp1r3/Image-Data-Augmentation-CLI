from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_aug_cli.config import AnnotationConfig, MaskConfig, YoloConfig


@dataclass(frozen=True)
class YoloLabels:
    exists: bool
    bboxes: list[list[float]]
    class_labels: list[str]


@dataclass(frozen=True)
class MaskData:
    exists: bool
    image: np.ndarray | None


def read_yolo_labels(
    src_path: Path,
    input_dir: Path,
    annotations: AnnotationConfig,
) -> YoloLabels:
    config = annotations.yolo
    label_path = build_input_annotation_path(
        src_path=src_path,
        input_dir=input_dir,
        root=config.labels_dir,
        extension=".txt",
    )
    if not label_path.exists():
        if config.allow_missing:
            return YoloLabels(exists=False, bboxes=[], class_labels=[])
        raise FileNotFoundError(f"YOLO label file was not found: {label_path}")

    bboxes: list[list[float]] = []
    class_labels: list[str] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row in {label_path}:{line_number}: {raw_line}")

        class_labels.append(parts[0])
        bboxes.append([float(value) for value in parts[1:]])

    return YoloLabels(exists=True, bboxes=bboxes, class_labels=class_labels)


def write_yolo_labels(
    src_path: Path,
    input_dir: Path,
    dst_image_path: Path,
    output_dir: Path,
    config: YoloConfig,
    bboxes: list[list[float]],
    class_labels: list[str],
) -> None:
    label_path = build_output_annotation_path(
        src_path=src_path,
        input_dir=input_dir,
        dst_image_path=dst_image_path,
        output_dir=output_dir,
        root=config.output_labels_dir,
        extension=".txt",
    )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
        for label, bbox in zip(class_labels, bboxes)
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_mask(src_path: Path, input_dir: Path, annotations: AnnotationConfig) -> MaskData:
    config = annotations.masks
    mask_path = build_input_annotation_path(
        src_path=src_path,
        input_dir=input_dir,
        root=config.masks_dir,
        extension=_normalize_extension(config.mask_extension),
    )
    if not mask_path.exists():
        if config.allow_missing:
            return MaskData(exists=False, image=None)
        raise FileNotFoundError(f"Segmentation mask was not found: {mask_path}")

    data = np.fromfile(str(mask_path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"OpenCV could not decode mask: {mask_path}")
    return MaskData(exists=True, image=mask)


def write_mask(
    src_path: Path,
    input_dir: Path,
    dst_image_path: Path,
    output_dir: Path,
    config: MaskConfig,
    mask: np.ndarray,
) -> None:
    mask_path = build_output_annotation_path(
        src_path=src_path,
        input_dir=input_dir,
        dst_image_path=dst_image_path,
        output_dir=output_dir,
        root=config.output_masks_dir,
        extension=_normalize_extension(config.mask_extension),
    )
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(mask_path.suffix, mask)
    if not success:
        raise ValueError(f"OpenCV could not encode output mask as {mask_path.suffix}.")
    encoded.tofile(str(mask_path))


def build_input_annotation_path(
    src_path: Path,
    input_dir: Path,
    root: str | None,
    extension: str,
) -> Path:
    relative = src_path.relative_to(input_dir).with_suffix(extension)
    if root is None:
        return src_path.with_suffix(extension)

    root_path = Path(root)
    if root_path.is_absolute():
        return root_path / relative
    return input_dir.parent / root_path / relative


def build_output_annotation_path(
    src_path: Path,
    input_dir: Path,
    dst_image_path: Path,
    output_dir: Path,
    root: str | None,
    extension: str,
) -> Path:
    if root is None:
        return dst_image_path.with_suffix(extension)

    relative_parent = src_path.relative_to(input_dir).parent
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path / relative_parent / f"{dst_image_path.stem}{extension}"
    return output_dir.parent / root_path / relative_parent / f"{dst_image_path.stem}{extension}"


def _normalize_extension(extension: str) -> str:
    normalized = extension.lower().strip()
    return normalized if normalized.startswith(".") else f".{normalized}"
