from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from image_aug_cli.config import ConfigError, load_config
from image_aug_cli.engine import run_augmentation


app = typer.Typer(
    add_completion=False,
    help="Mass image augmentation CLI powered by Albumentations.",
)
console = Console()


@app.command()
def augment(
    input_dir: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory with raw images.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory where augmented dataset will be saved.",
    ),
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="YAML or JSON augmentation config.",
    ),
) -> None:
    """Apply an augmentation pipeline to images recursively."""
    try:
        config = load_config(config_path)
        summary = run_augmentation(input_dir=input_dir, output_dir=output_dir, config=config)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(
        "[green]done[/green] "
        f"discovered={summary.discovered} "
        f"written={summary.written} "
        f"skipped={summary.skipped} "
        f"failed={summary.failed}"
    )
    if summary.failed:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
