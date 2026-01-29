import os
import random
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw


# =========================
# CONFIGURATION
# =========================

DIV2K_ROOT = "archive"     # contains DIV2K_train_HR and DIV2K_valid_HR
OUTPUT_ROOT = "prepared_dataset"

IMAGE_SIZE = 256
NUM_SAMPLES = 5000

MASK_STROKES = (5, 12)
MASK_THICKNESS = (4, 8)


# =========================
# UTILS
# =========================

def list_all_images_recursive(root: str) -> List[str]:
    """
    Recursively finds all image files under root.
    Works with nested DIV2K directories.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(dirpath, f))
    return paths


def random_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        scale = max(size / w, size / h)
        img = img.resize(
            (int(w * scale), int(h * scale)), Image.BICUBIC
        )
        w, h = img.size

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


def generate_random_mask(
    size: int,
    strokes: Tuple[int, int],
    thickness: Tuple[int, int],
) -> Image.Image:
    """
    Continuous random stroke mask with CONSTANT thickness.
    255 = valid pixel
    0   = hole
    """
    mask = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask)

    num_strokes = random.randint(*strokes)

    # one constant thickness per mask
    stroke_width = random.randint(*thickness)

    # random starting point
    x, y = random.randint(0, size - 1), random.randint(0, size - 1)

    for _ in range(num_strokes):
        # random direction and length
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(size // 8, size // 3)

        x_new = int(x + length * np.cos(angle))
        y_new = int(y + length * np.sin(angle))

        # keep inside image bounds
        x_new = max(0, min(size - 1, x_new))
        y_new = max(0, min(size - 1, y_new))

        draw.line(
            (x, y, x_new, y_new),
            fill=0,
            width=stroke_width,
        )

        # continue from previous end
        x, y = x_new, y_new

    return mask


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    img = np.array(image).astype(np.float32)
    m = (np.array(mask) > 0).astype(np.float32)
    masked = img * m[..., None]
    return Image.fromarray(masked.astype(np.uint8))


def save_visualization(
    original: Image.Image,
    masked: Image.Image,
    mask: Image.Image,
    path: str,
):
    w, h = original.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(original, (0, 0))
    canvas.paste(masked, (w, 0))
    canvas.paste(mask.convert("RGB"), (w * 2, 0))
    canvas.save(path)


# =========================
# MAIN
# =========================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    dirs = {
        "original": os.path.join(OUTPUT_ROOT, "original"),
        "masked": os.path.join(OUTPUT_ROOT, "masked"),
        "mask": os.path.join(OUTPUT_ROOT, "mask"),
        "visualization": os.path.join(OUTPUT_ROOT, "visualization"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    image_paths = list_all_images_recursive(DIV2K_ROOT)
    assert len(image_paths) > 0, "No DIV2K images found!"

    print(f"Found {len(image_paths)} DIV2K images (train + valid)")

    for i in range(NUM_SAMPLES):
        img_path = random.choice(image_paths)
        img = Image.open(img_path).convert("RGB")

        img_crop = random_crop(img, IMAGE_SIZE)
        mask = generate_random_mask(
            IMAGE_SIZE, MASK_STROKES, MASK_THICKNESS
        )
        masked = apply_mask(img_crop, mask)

        img_crop.save(os.path.join(dirs["original"], f"{i:05d}.png"))
        masked.save(os.path.join(dirs["masked"], f"{i:05d}.png"))
        mask.save(os.path.join(dirs["mask"], f"{i:05d}.png"))

        save_visualization(
            img_crop,
            masked,
            mask,
            os.path.join(dirs["visualization"], f"{i:05d}.png"),
        )

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES} samples")

    print("\nDataset preparation finished successfully.")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
