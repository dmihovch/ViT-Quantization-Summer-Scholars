"""
download_imagenet_val.py
========================

Download ImageNet-1K **validation** images from Hugging Face into `data/`.

Why streaming?
--------------
The full ImageNet validation split is 50,000 images (~6.7 GB). We almost never
need all of them at once, so this script *streams* the dataset and stops after
saving the number of images you ask for. Nothing is cached to disk beyond the
images themselves.

On-disk layout
--------------
Images are written in the standard "ImageFolder" layout, one folder per class:

    data/
      class_000/val_00000.jpeg
      class_000/val_00001.jpeg
      class_217/val_00002.jpeg
      ...

The folder name encodes the integer ImageNet label, so the class information is
preserved for the later accuracy experiments. Experiment 1 ignores labels and
just globs every image under `data/`, so this layout works for it too.

A note on access
----------------
The canonical dataset (`ILSVRC/imagenet-1k`) is *gated*: you must have a free
Hugging Face account, accept the dataset's terms on its page, and log in once
with `hf auth login`. If you hit an access error, the script prints exactly
these steps. You can also point `--dataset` at any non-gated mirror that exposes
an `image` column and (optionally) a `label` column.

Examples
--------
    # Small set for quick debugging runs.
    python download_imagenet_val.py --num-images 128

    # The default characterization set.
    python download_imagenet_val.py --num-images 4096

    # The full validation split for a final thesis-print run.
    python download_imagenet_val.py --num-images 50000
"""

import argparse
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# The official ImageNet-1K dataset on the Hugging Face Hub (gated - see above).
DEFAULT_DATASET: str = "ILSVRC/imagenet-1k"
DEFAULT_SPLIT: str = "validation"


@dataclass(frozen=True)
class DownloadConfig:
    """Every knob for one download, gathered into a single immutable object."""

    dataset_id: str
    split: str
    num_images: int
    output_dir: Path
    image_column: str
    label_column: str


def parse_config() -> DownloadConfig:
    """Read command-line arguments into the immutable `DownloadConfig`."""
    parser = argparse.ArgumentParser(
        description="Download ImageNet-1K validation images from Hugging Face into data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4096,
        help="How many validation images to download (e.g. 128, 4096, 50000).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id to pull from.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Which split to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Folder to save images into (created if missing).",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image",
        help="Name of the column holding the image in the dataset.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of the column holding the integer class label.",
    )
    arguments = parser.parse_args()

    return DownloadConfig(
        dataset_id=arguments.dataset,
        split=arguments.split,
        num_images=arguments.num_images,
        output_dir=arguments.output_dir,
        image_column=arguments.image_column,
        label_column=arguments.label_column,
    )


def coerce_to_pil_image(raw_image: object) -> Image.Image:
    """
    Normalize whatever the dataset hands us into a PIL image.

    With the default settings, Hugging Face already decodes the `image` column
    into a `PIL.Image`. As a fallback we also handle the undecoded form, which
    is a dict containing the raw file `bytes`.
    """
    if isinstance(raw_image, Image.Image):
        return raw_image
    if isinstance(raw_image, dict) and isinstance(raw_image.get("bytes"), bytes):
        return Image.open(io.BytesIO(raw_image["bytes"]))
    raise TypeError(f"Could not interpret image of type {type(raw_image)!r}.")


def coerce_to_label(raw_label: object) -> int | None:
    """Return the integer class label, or None if the dataset has no labels."""
    if isinstance(raw_label, bool):
        return None  # guard: bool is a subclass of int but is not a class label
    if isinstance(raw_label, int):
        return raw_label
    return None


def save_image(
    image: Image.Image, label: int | None, index: int, output_dir: Path
) -> None:
    """
    Save one image as a JPEG inside a per-class subfolder of `output_dir`.

    Images are converted to RGB so that grayscale or CMYK source images come out
    with exactly 3 channels (what the model expects).
    """
    class_folder = f"class_{label:03d}" if label is not None else "unlabeled"
    class_dir = output_dir / class_folder
    class_dir.mkdir(parents=True, exist_ok=True)

    rgb_image = image.convert("RGB")
    destination = class_dir / f"val_{index:05d}.jpeg"
    rgb_image.save(destination, format="JPEG", quality=95)


def download_images(config: DownloadConfig) -> int:
    """
    Stream the dataset and save up to `config.num_images` images.

    Returns the number of images actually saved. The heavy `datasets` import
    happens here (not at module top) so the rest of this file stays importable
    without it.
    """
    from datasets import load_dataset

    print(f"Streaming '{config.dataset_id}' split '{config.split}' ...")
    dataset = load_dataset(config.dataset_id, split=config.split, streaming=True)

    saved_count = 0
    for example in dataset:
        if saved_count >= config.num_images:
            break

        image = coerce_to_pil_image(example[config.image_column])
        label = coerce_to_label(example.get(config.label_column))
        save_image(image, label, saved_count, config.output_dir)

        saved_count += 1
        if saved_count % 100 == 0 or saved_count == config.num_images:
            print(f"  saved {saved_count} / {config.num_images} images", end="\r")

    print()  # finish the in-place progress line
    return saved_count


def main() -> None:
    config = parse_config()
    try:
        saved_count = download_images(config)
    except Exception as error:
        # The most common failure is the dataset being gated and the user not
        # being logged in. Give actionable next steps instead of a raw traceback.
        print("\nFailed to download from Hugging Face.")
        print(f"  Reason: {error}\n")
        print("If the dataset is gated (e.g. ILSVRC/imagenet-1k) you must:")
        print("  1. Create a free account at https://huggingface.co")
        print("  2. Accept the dataset's terms on its page.")
        print("  3. Log in once on this machine:  hf auth login")
        print("Alternatively, pass --dataset <id> to use a non-gated mirror.")
        raise SystemExit(1) from error

    print(f"Done. Saved {saved_count} images under '{config.output_dir}'.")

    # Hugging Face's streaming stack spawns background threads (aiohttp / pyarrow)
    # that can crash the interpreter during normal shutdown. Our work is finished
    # and every image has already been flushed to disk, so we flush our output
    # and exit immediately, bypassing that buggy finalization entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
