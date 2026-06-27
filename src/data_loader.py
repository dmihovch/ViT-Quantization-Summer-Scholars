"""
data_loader.py
==============

Loads the images we push through the model in Experiment 1.

Outlier *mapping* does not need class labels - we only run forward passes to
observe activations - so we use a deliberately simple dataset that just reads
image files from a directory and applies the model's preprocessing transform.

Expected data layout
---------------------
Point `image_dir` at any folder containing image files; subfolders are searched
recursively. A typical ImageNet validation split looks like:

    data/
      n01440764/ILSVRC2012_val_00000293.JPEG
      n01443537/ILSVRC2012_val_00000236.JPEG
      ...

but a flat folder of images works equally well, because we ignore labels.
"""

from pathlib import Path
from typing import override

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.model_utils import ImageTransform

# Image file types we will pick up from the data directory.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def find_image_files(image_dir: Path) -> list[Path]:
    """
    Recursively collect every image file under `image_dir`, sorted by path so
    that repeated runs see the images in the same deterministic order.
    """
    matching_paths: list[Path] = []
    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            matching_paths.append(path)
    return matching_paths


class UnlabeledImageDataset(Dataset[Tensor]):
    """
    A minimal image dataset mapping: index -> preprocessed image tensor.

    Each item is a float tensor of shape [3, 224, 224] (channels, height, width)
    after the ViT preprocessing transform has resized, cropped, and normalized
    the source image.
    """

    def __init__(
        self, image_dir: Path, transform: ImageTransform, max_images: int
    ) -> None:
        # Keep at most `max_images` files (Experiment 1 uses 4,096).
        self.image_paths: list[Path] = find_image_files(image_dir)[:max_images]
        self.transform: ImageTransform = transform

        if len(self.image_paths) == 0:
            message: str = (
                f"No images with extensions {IMAGE_EXTENSIONS} were found under "
                + f"'{image_dir}'. Place the ImageNet validation images there."
            )
            raise FileNotFoundError(message)

    def __len__(self) -> int:
        return len(self.image_paths)

    @override
    def __getitem__(self, index: int) -> Tensor:
        image_path: Path = self.image_paths[index]
        # Convert to RGB so that grayscale or CMYK images still come out with
        # exactly 3 channels, matching what the model expects.
        image: Image.Image = Image.open(image_path).convert("RGB")
        return self.transform(image)


def build_image_dataloader(
    image_dir: Path,
    transform: ImageTransform,
    batch_size: int,
    max_images: int,
) -> DataLoader[Tensor]:
    """
    Wrap `UnlabeledImageDataset` in a DataLoader that yields batches of shape
    [Batch, 3, 224, 224].

    `shuffle=False` keeps the image order deterministic, which makes the
    measured statistics reproducible from one run to the next.
    """
    dataset = UnlabeledImageDataset(image_dir, transform, max_images)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # decode and transform images on background CPU workers
        pin_memory=True,  # use page-locked memory for faster host -> GPU copies
    )
