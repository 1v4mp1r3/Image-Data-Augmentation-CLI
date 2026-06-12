# Image Augmentation CLI

High-throughput starter CLI for recursively augmenting image datasets while preserving the
input folder structure.

## Architecture

- `image_aug_cli/cli.py` owns the command-line surface and exit codes.
- `image_aug_cli/config.py` loads YAML/JSON into small validated dataclasses.
- `image_aug_cli/pipeline.py` converts config transform specs into an Albumentations
  `Compose` pipeline.
- `image_aug_cli/io.py` handles recursive discovery plus OpenCV read/write helpers.
- `image_aug_cli/engine.py` coordinates the multiprocessing pool, progress bar, and per-image
  worker execution.

This split keeps the CLI thin, the config schema testable, and the augmentation engine usable
from Python later.

## Usage

```powershell
pip install -e .
imgaug --input C:\datasets\raw --output C:\datasets\augmented --config configs\example.yaml
```

On Windows, you can use the helper scripts from the project root:

```bat
install_requirements.bat
run_imgaug.bat --input C:\datasets\raw --output C:\datasets\augmented --config configs\example.yaml
```

`C:\datasets\raw` is only an example path. Replace it with a real folder that contains
your images. To run without activating `.venv` manually, use `imgaug.bat`:

```bat
imgaug.bat --help
imgaug.bat --input C:\path\to\images --output C:\path\to\augmented --config configs\example.yaml
```

The example config creates three augmented images per source image and keeps subdirectories:

```text
raw\cats\a.jpg -> augmented\cats\a_aug000.jpeg
raw\cats\a.jpg -> augmented\cats\a_aug001.jpeg
raw\cats\a.jpg -> augmented\cats\a_aug002.jpeg
```

## YAML Config

See `configs/example.yaml` for a ready-to-run pipeline with resize, flip, rotate,
brightness/contrast, blur, Gaussian noise, and salt-and-pepper noise.

## Object Detection Roadmap

For YOLO support, add an annotation adapter layer before the worker writes output:

1. Read `image.jpg` plus `image.txt`.
2. Convert YOLO boxes to the format Albumentations expects.
3. Build `A.Compose(..., bbox_params=A.BboxParams(coord_format="yolo", label_fields=["class_labels"]))`.
4. Call the same pipeline with `image=image, bboxes=bboxes, class_labels=labels`.
5. Drop invalid boxes according to policy and write the transformed YOLO file beside the image.

The important rule is that geometry transforms must see image and annotations in the same call.
