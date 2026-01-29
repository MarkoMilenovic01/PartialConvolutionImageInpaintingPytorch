import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model import PartialUNet


# =========================
# CONFIGURATION
# =========================

DATASET_ROOT = "prepared_dataset"
OUTPUT_DIR = "training_output"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4


# =========================
# DATASET
# =========================

class InpaintingDataset(Dataset):
    def __init__(self, root_dir):
        self.original_dir = os.path.join(root_dir, "original")
        self.masked_dir = os.path.join(root_dir, "masked")
        self.mask_dir = os.path.join(root_dir, "mask")

        self.files = sorted(os.listdir(self.original_dir))

        self.img_transform = T.ToTensor()   # [0,1]
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

        original = self.img_transform(original)   # (3,H,W)
        masked = self.img_transform(masked)       # (3,H,W)
        mask = self.mask_transform(mask)          # (1,H,W)

        mask = (mask > 0).float()                  # binary mask

        return original, masked, mask


# =========================
# MASKED L1 LOSS
# =========================

def masked_l1_loss(predicted, target, mask):
    hole = 1.0 - mask
    return ((predicted - target).abs() * hole).sum() / (hole.sum() + 1e-8)


# =========================
# TRAINING
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = InpaintingDataset(DATASET_ROOT)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = PartialUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []

    print(f"Training on {device}")
    print(f"Samples: {len(dataset)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for img, masked, mask in dataloader:
            img = img.to(device)
            masked = masked.to(device)
            mask = mask.to(device)

            pred = model(masked, mask)

            loss = masked_l1_loss(pred, img, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)

        print(f"Epoch [{epoch:03d}/{EPOCHS}]  Masked L1 loss: {epoch_loss:.6f}")

    # =========================
    # SAVE MODEL
    # =========================

    model_path = os.path.join(OUTPUT_DIR, "partial_unet.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # =========================
    # PLOT LOSS
    # =========================

    plt.figure()
    plt.plot(losses, label="Masked L1 loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss curve")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(OUTPUT_DIR, "training_loss.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    main()
