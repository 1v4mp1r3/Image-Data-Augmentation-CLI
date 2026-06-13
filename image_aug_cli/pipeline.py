from __future__ import annotations

from typing import Any

import albumentations as A

from image_aug_cli.config import AugmentConfig, ConfigError


ALLOWED_TRANSFORMS = {
    "Resize",
    "RandomCrop",
    "CenterCrop",
    "RandomResizedCrop",
    "HorizontalFlip",
    "VerticalFlip",
    "Transpose",
    "RandomRotate90",
    "Rotate",
    "Affine",
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion",
    "GaussianBlur",
    "MedianBlur",
    "MotionBlur",
    "GaussNoise",
    "SaltAndPepper",
    "RandomBrightnessContrast",
    "RandomGamma",
    "CLAHE",
    "ColorJitter",
    "HueSaturationValue",
    "Sharpen",
}


def build_pipeline(config: AugmentConfig) -> A.Compose:
    transforms = []
    for spec in config.transforms:
        if spec.name not in ALLOWED_TRANSFORMS:
            allowed = ", ".join(sorted(ALLOWED_TRANSFORMS))
            raise ConfigError(f"Unsupported transform '{spec.name}'. Allowed transforms: {allowed}")

        transform_cls = getattr(A, spec.name, None)
        if transform_cls is None:
            raise ConfigError(
                f"Albumentations does not expose transform '{spec.name}'. "
                "Check the installed albumentations version."
            )

        params = _normalize_params(spec.params)
        try:
            transforms.append(transform_cls(**params))
        except Exception as exc:  # Albumentations raises pydantic/value errors here.
            raise ConfigError(f"Invalid parameters for transform '{spec.name}': {exc}") from exc

    bbox_params = None
    if config.annotations.yolo.enabled:
        bbox_params = A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=config.annotations.yolo.min_area,
            min_visibility=config.annotations.yolo.min_visibility,
            clip=config.annotations.yolo.clip,
            filter_invalid_bboxes=config.annotations.yolo.filter_invalid_bboxes,
        )

    return A.Compose(transforms, bbox_params=bbox_params, strict=True)


def _normalize_params(value: Any) -> Any:
    """Turn YAML lists into tuples for Albumentations range parameters."""
    if isinstance(value, list):
        return tuple(_normalize_params(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_params(item) for key, item in value.items()}
    return value
