from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from image_aug_cli.config import AugmentConfig
from image_aug_cli.io import build_output_path, iter_images, read_rgb_image, write_rgb_image
from image_aug_cli.pipeline import build_pipeline


_WORKER_CONFIG: AugmentConfig | None = None
_WORKER_PIPELINE: Any | None = None


@dataclass(frozen=True)
class RunSummary:
    discovered: int
    written: int
    skipped: int
    failed: int


@dataclass(frozen=True)
class ImageResult:
    source: str
    written: int = 0
    skipped: int = 0
    error: str | None = None


def run_augmentation(input_dir: Path, output_dir: Path, config: AugmentConfig) -> RunSummary:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    images = iter_images(input_dir, config.extensions)

    if not images:
        return RunSummary(discovered=0, written=0, skipped=0, failed=0)

    workers = config.runtime.workers or os.cpu_count() or 1
    workers = max(1, workers)
    chunksize = max(1, config.runtime.chunksize)
    tasks = [(str(path), str(input_dir), str(output_dir)) for path in images]
    build_pipeline(config)

    progress = Progress(
        TextColumn("[bold]augmenting"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("elapsed"),
        TextColumn("|"),
        TimeRemainingColumn(),
    )

    written = skipped = failed = 0
    with progress:
        task_id = progress.add_task("augment", total=len(tasks))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(config,),
        ) as executor:
            results = executor.map(_process_one_image, tasks, chunksize=chunksize)

            for result in results:
                written += result.written
                skipped += result.skipped
                failed += 1 if result.error else 0
                if result.error:
                    progress.console.print(
                        f"[red]failed[/red] {result.source}: {result.error}"
                    )
                progress.advance(task_id)

    return RunSummary(
        discovered=len(images),
        written=written,
        skipped=skipped,
        failed=failed,
    )


def _init_worker(config: AugmentConfig) -> None:
    import cv2

    global _WORKER_CONFIG, _WORKER_PIPELINE
    cv2.setNumThreads(1)
    _WORKER_CONFIG = config
    _WORKER_PIPELINE = build_pipeline(config)


def _process_one_image(task: tuple[str, str, str]) -> ImageResult:
    if _WORKER_CONFIG is None or _WORKER_PIPELINE is None:
        raise RuntimeError("Worker was not initialized.")

    src_path = Path(task[0])
    input_dir = Path(task[1])
    output_dir = Path(task[2])
    config = _WORKER_CONFIG

    try:
        image = read_rgb_image(src_path)
        written = skipped = 0

        for index in range(config.output.samples_per_image):
            dst_path = build_output_path(
                src_path=src_path,
                input_dir=input_dir,
                output_dir=output_dir,
                index=index,
                output=config.output,
            )
            if dst_path.exists() and not config.output.overwrite:
                skipped += 1
                continue

            result = _WORKER_PIPELINE(image=image)
            write_rgb_image(dst_path, result["image"], config.output)
            written += 1

        return ImageResult(source=str(src_path), written=written, skipped=skipped)
    except Exception as exc:
        return ImageResult(source=str(src_path), error=str(exc))
