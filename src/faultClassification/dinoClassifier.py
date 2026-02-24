"""
Manufacturing Fault Classifier using DINOv2 + Albumentations
- Aspect-ratio-preserving resize with padding
- Random resized crops + flips for training augmentation
- DINOv2 ViT-B/14 as frozen backbone + trainable classification head
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG = {
    "image_size": 224,          # DINOv2 default input size (must be divisible by patch size 14)
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # DINOv2 ImageNet normalization stats
    "mean": (0.485, 0.456, 0.406),
    "std":  (0.229, 0.224, 0.225),
}


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def make_padding_transform(image_size: int, mean, std) -> A.Compose:
    """
    Aspect-ratio-preserving resize + center padding.
    Used for both validation and as the base for training augmentation.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),          # resize longest edge to target size
        A.PadIfNeeded(                                   # pad shorter edge to make it square
            min_height=image_size,
            min_width=image_size,
            border_mode=0,                               # constant (black) padding
            fill=0,
        ),
    ])


def get_train_transform(image_size: int, mean, std) -> A.Compose:
    """
    Training pipeline:
      1. Resize with aspect-ratio-preserving padding (slightly oversized)
      2. Random resized crop to target size  →  scale robustness
      3. Flips + light color jitter          →  appearance robustness
      4. Normalize + ToTensor
    """
    return A.Compose([
        # Step 1: pad to slightly larger than target so random crop has room to work
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(
            min_height=int(image_size * 1.15),
            min_width=int(image_size * 1.15),
            border_mode=0,
            fill=0,
        ),

        # Step 2: random resized crop — teaches scale invariance
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),       # crop between 70–100% of the padded image area
            ratio=(0.85, 1.15),     # allow slight aspect ratio variation
        ),

        # Step 3: geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.4),

        # Step 4: color/appearance augmentations (keep subtle for industrial images)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),

        # Step 5: normalize and convert to tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int, mean, std) -> A.Compose:
    """
    Validation pipeline: pad-preserve aspect ratio, then normalize.
    No randomness.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=0,
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class FaultDataset(Dataset):
    """
    Expects a directory structure:
        root/
            fault_type_a/
                image1.jpg
                image2.png
                ...
            fault_type_b/
                ...

    Or alternatively, pass lists of image paths and labels directly.
    """

    def __init__(self, image_paths: list, labels: list, transform: A.Compose):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]
        augmented = self.transform(image=image)
        return augmented["image"], label

    @classmethod
    def from_directory(cls, root_dir: str, transform: A.Compose, label_encoder: LabelEncoder = None):
        """
        Build dataset from a class-per-folder directory structure.
        Returns (dataset, label_encoder) so the encoder can be reused for val/test sets.
        """
        root = Path(root_dir)
        image_paths, raw_labels = [], []

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    image_paths.append(str(img_path))
                    raw_labels.append(class_dir.name)

        if label_encoder is None:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(raw_labels)
        else:
            encoded_labels = label_encoder.transform(raw_labels)

        return cls(image_paths, encoded_labels.tolist(), transform), label_encoder


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-B/14 as a frozen feature extractor with a trainable MLP head.
    Only the classification head is updated during training, keeping GPU memory
    and training time low — ideal for small datasets.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        # Load pretrained DINOv2 backbone from torch.hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        backbone_out_dim = self.backbone.embed_dim  # 768 for ViT-B

        # Trainable classification head
        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)          # [B, 768]
        return self.head(features)               # [B, num_classes]

    def unfreeze_backbone(self, num_layers: int = 2):
        """
        Optional: unfreeze the last N transformer blocks for fine-tuning
        after initial head training converges. Call after ~10 epochs.
        """
        blocks = list(self.backbone.blocks)
        for block in blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfroze last {num_layers} transformer blocks.")


# ─────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(train_dir: str, val_dir: str, output_dir: str = "./checkpoints"):
    cfg = CONFIG
    device = cfg["device"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # Transforms
    train_transform = get_train_transform(cfg["image_size"], cfg["mean"], cfg["std"])
    val_transform   = get_val_transform(cfg["image_size"], cfg["mean"], cfg["std"])

    # Datasets
    train_dataset, label_encoder = FaultDataset.from_directory(train_dir, train_transform)
    val_dataset, _               = FaultDataset.from_directory(val_dir, val_transform, label_encoder)

    num_classes = len(label_encoder.classes_)
    print(f"Classes ({num_classes}): {list(label_encoder.classes_)}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    model = DINOv2Classifier(num_classes=num_classes).to(device)

    # Loss — use label smoothing to reduce overfitting on small datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only optimize the head parameters initially
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])
    scaler = torch.GradScaler()

    best_val_acc = 0.0

    for epoch in range(cfg["num_epochs"]):

        # Optional: unfreeze last 2 backbone blocks halfway through training
        if epoch == cfg["num_epochs"] // 2:
            model.unfreeze_backbone(num_layers=2)
            # Re-initialize optimizer to include newly unfrozen params with lower LR
            optimizer = AdamW([
                {"params": model.head.parameters(),     "lr": cfg["learning_rate"]},
                {"params": filter(lambda p: p.requires_grad,
                                  model.backbone.parameters()), "lr": cfg["learning_rate"] * 0.1},
            ], weight_decay=cfg["weight_decay"])
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"] // 2)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch+1:>3}/{cfg['num_epochs']}] "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "label_encoder_classes": label_encoder.classes_,
                "val_acc": val_acc,
                "config": cfg,
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    # Final evaluation report
    print("\n── Final Validation Report ──")
    print(classification_report(val_labels, val_preds, target_names=label_encoder.classes_))
    print(f"Best val accuracy: {best_val_acc:.3f}")


# ─────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────

def load_model_for_inference(checkpoint_path: str) -> tuple[DINOv2Classifier, LabelEncoder, dict]:
    """Load a saved checkpoint and return the model ready for inference."""
    cfg = CONFIG
    checkpoint = torch.load(checkpoint_path, map_location=cfg["device"])

    label_encoder = LabelEncoder()
    label_encoder.classes_ = checkpoint["label_encoder_classes"]
    num_classes = len(label_encoder.classes_)

    model = DINOv2Classifier(num_classes=num_classes).to(cfg["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, label_encoder, cfg


@torch.no_grad()
def predict(model: DINOv2Classifier, image_path: str, label_encoder: LabelEncoder, cfg: dict) -> dict:
    """Run inference on a single image and return class probabilities."""
    transform = get_val_transform(cfg["image_size"], cfg["mean"], cfg["std"])
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=image)["image"].unsqueeze(0).to(cfg["device"])

    with torch.autocast(device_type=cfg["device"], dtype=torch.float16):
        logits = model(tensor)

    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx = probs.argmax()

    return {
        "predicted_class": label_encoder.classes_[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probabilities": {cls: float(p) for cls, p in zip(label_encoder.classes_, probs)},
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Replace these paths with your actual data directories ──
    # Expected structure:
    #   data/train/scratch/img1.jpg, img2.jpg, ...
    #   data/train/crack/img1.jpg, ...
    #   data/val/scratch/img1.jpg, ...
    #   data/val/crack/img1.jpg, ...

    train(
        train_dir="data/train",
        val_dir="data/val",
        output_dir="checkpoints",
    )