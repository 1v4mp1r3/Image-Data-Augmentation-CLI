from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class ConfigError(ValueError):
    """Raised when a config file cannot be parsed or validated."""


@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    samples_per_image: int = 1
    format: str | None = None
    suffix: str = "_aug{index:03d}"
    preserve_tree: bool = True
    overwrite: bool = False
    jpeg_quality: int = 95
    png_compression: int = 3


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int | None = None
    chunksize: int = 16


@dataclass(frozen=True)
class AugmentConfig:
    transforms: tuple[TransformSpec, ...]
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AugmentConfig":
        if not isinstance(raw, dict):
            raise ConfigError("Config root must be a mapping.")

        raw_transforms = raw.get("transforms")
        if not isinstance(raw_transforms, list) or not raw_transforms:
            raise ConfigError("Config must contain a non-empty 'transforms' list.")

        transforms = tuple(_parse_transform(item) for item in raw_transforms)
        output = _parse_dataclass(OutputConfig, raw.get("output", {}), "output")
        runtime = _parse_dataclass(RuntimeConfig, raw.get("runtime", {}), "runtime")
        _validate_output(output)
        _validate_runtime(runtime)

        extensions = raw.get("extensions", DEFAULT_EXTENSIONS)
        if not isinstance(extensions, (list, tuple)) or not extensions:
            raise ConfigError("'extensions' must be a non-empty list.")

        normalized_extensions = tuple(
            ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
            for ext in extensions
        )

        return cls(
            transforms=transforms,
            output=output,
            runtime=runtime,
            extensions=normalized_extensions,
        )


def load_config(path: Path) -> AugmentConfig:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc

    try:
        if suffix == ".json":
            raw = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        else:
            raise ConfigError("Config format must be .yaml, .yml, or .json.")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot parse config file {path}: {exc}") from exc

    return AugmentConfig.from_mapping(raw)


def _parse_transform(raw: Any) -> TransformSpec:
    if not isinstance(raw, dict):
        raise ConfigError("Each transform must be a mapping.")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("Each transform must contain a non-empty 'name'.")

    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError(f"Transform '{name}' has invalid 'params'; expected a mapping.")

    return TransformSpec(name=name, params=params)


def _parse_dataclass(cls: type[Any], raw: Any, section: str) -> Any:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"'{section}' must be a mapping.")

    allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw) - allowed
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown keys in '{section}': {unknown_list}")

    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"Invalid '{section}' section: {exc}") from exc


def _validate_output(output: OutputConfig) -> None:
    if output.samples_per_image < 1:
        raise ConfigError("'output.samples_per_image' must be at least 1.")
    if output.jpeg_quality < 1 or output.jpeg_quality > 100:
        raise ConfigError("'output.jpeg_quality' must be between 1 and 100.")
    if output.png_compression < 0 or output.png_compression > 9:
        raise ConfigError("'output.png_compression' must be between 0 and 9.")
    if "{index" not in output.suffix and output.samples_per_image > 1:
        raise ConfigError(
            "'output.suffix' must include '{index}' when samples_per_image is greater than 1."
        )


def _validate_runtime(runtime: RuntimeConfig) -> None:
    if runtime.workers is not None and runtime.workers < 1:
        raise ConfigError("'runtime.workers' must be at least 1.")
    if runtime.chunksize < 1:
        raise ConfigError("'runtime.chunksize' must be at least 1.")
