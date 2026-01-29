import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np

from skimage.metrics import structural_similarity as ssim
from model import PartialUNet


# =========================
# CONFIGURATION
# =========================

DATASET_ROOT = "prepared_dataset"
MODEL_PATH = "training_output/partial_unet.pth"
OUTPUT_DIR = "evaluation_output"

BATCH_SIZE = 32
NUM_VISUAL = 50
NUM_EVAL = 1000


# =========================
# DATASET (same as training)
# =========================

class InpaintingDataset(Dataset):
    def __init__(self, root_dir):
        self.original_dir = os.path.join(root_dir, "original")
        self.masked_dir = os.path.join(root_dir, "masked")
        self.mask_dir = os.path.join(root_dir, "mask")

        self.files = sorted(os.listdir(self.original_dir))

        self.img_transform = T.ToTensor()
        self.mask_transform = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        original = Image.open(
            os.path.join(self.original_dir, fname)
        ).convert("RGB")

        masked = Image.open(
            os.path.join(self.masked_dir, fname)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.mask_dir, fname)
        ).convert("L")

        original = self.img_transform(original)
        masked = self.img_transform(masked)
        mask = (self.mask_transform(mask) > 0).float()

        return original, masked, mask


# =========================
# METRICS (WHOLE IMAGE)
# =========================

def l1_error(pred, gt):
    return torch.mean(torch.abs(pred - gt)).item()


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def ssim_metric(pred, gt):
    pred = pred.detach().cpu().permute(1, 2, 0).numpy()
    gt = gt.detach().cpu().permute(1, 2, 0).numpy()
    return ssim(gt, pred, data_range=1.0, channel_axis=2)


# =========================
# VISUALIZATION
# =========================

def save_visualization(idx, original, masked, mask, output):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def to_img(x):
        x = x.detach().cpu().clamp(0, 1)
        x = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(x)

    orig_img = to_img(original)
    masked_img = to_img(masked)
    out_img = to_img(output)

    mask_img = (mask[0].detach().cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_img).convert("RGB")

    w, h = orig_img.size
    canvas = Image.new("RGB", (w * 2, h * 2))

    canvas.paste(orig_img, (0, 0))
    canvas.paste(masked_img, (w, 0))
    canvas.paste(mask_img, (0, h))
    canvas.paste(out_img, (w, h))

    canvas.save(os.path.join(OUTPUT_DIR, f"visual_{idx:02d}.png"))


# =========================
# MAIN
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = InpaintingDataset(DATASET_ROOT)

    model = PartialUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Evaluation on device: {device}")

    # =========================
    # VISUAL EVALUATION
    # =========================

    with torch.no_grad():
        for i in range(NUM_VISUAL):
            original, masked, mask = dataset[i]

            original = original.to(device)
            masked = masked.to(device)
            mask = mask.to(device)

            output = model(masked.unsqueeze(0), mask.unsqueeze(0))[0]

            save_visualization(i, original, masked, mask, output)

    print(f"Saved {NUM_VISUAL} visualization images.")

    # =========================
    # STATISTICAL EVALUATION
    # =========================

    eval_dataset = torch.utils.data.Subset(dataset, list(range(NUM_EVAL)))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    l1_vals, psnr_vals, ssim_vals = [], [], []

    with torch.no_grad():
        for original, masked, mask in eval_loader:
            original = original.to(device)
            masked = masked.to(device)
            mask = mask.to(device)

            output = model(masked, mask)

            for i in range(output.size(0)):
                l1_vals.append(l1_error(output[i], original[i]))
                psnr_vals.append(psnr(output[i], original[i]))
                ssim_vals.append(ssim_metric(output[i], original[i]))

    # =========================
    # RESULTS
    # =========================

    def mean_std(x):
        return np.mean(x), np.std(x)

    l1_m, l1_s = mean_std(l1_vals)
    psnr_m, psnr_s = mean_std(psnr_vals)
    ssim_m, ssim_s = mean_std(ssim_vals)

    print("\nEvaluation results (whole image):")
    print(f"L1   : mean={l1_m:.6f}, std={l1_s:.6f}")
    print(f"PSNR : mean={psnr_m:.2f} dB, std={psnr_s:.2f}")
    print(f"SSIM : mean={ssim_m:.4f}, std={ssim_s:.4f}")

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write("Evaluation metrics (whole image)\n")
        f.write(f"L1   mean={l1_m:.6f}, std={l1_s:.6f}\n")
        f.write(f"PSNR mean={psnr_m:.2f} dB, std={psnr_s:.2f}\n")
        f.write(f"SSIM mean={ssim_m:.4f}, std={ssim_s:.4f}\n")

    print("Metrics saved to evaluation_output/metrics.txt")


if __name__ == "__main__":
    main()
