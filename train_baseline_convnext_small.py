#!/usr/bin/env python3

from __future__ import annotations
import argparse
import os
import random
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    import timm
    _HAVE_TIMM = True
except Exception:
    timm = None
    _HAVE_TIMM = False

try:
    from torchvision import transforms
    from torchvision.models import resnet18, resnet34, densenet121
    _HAVE_TORCHVISION = True
except Exception:
    transforms = None
    resnet18 = resnet34 = densenet121 = None
    _HAVE_TORCHVISION = False


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_path_col(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"--path-col '{user_col}' not found in CSV columns: {list(df.columns)}")
        return user_col
    for c in ["path", "filepath", "file_path", "image_path", "relpath", "relative_path"]:
        if c in df.columns:
            return c
    raise ValueError(
        "CSV must contain a path column. Looked for "
        f"path/filepath/file_path/image_path/relpath. Found: {list(df.columns)}"
    )


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


class HEStainJitter:
    def __init__(
        self,
        p: float = 0.5,
        h_scale: Tuple[float, float] = (0.9, 1.1),
        e_scale: Tuple[float, float] = (0.9, 1.1),
        h_shift: Tuple[float, float] = (-0.05, 0.05),
        e_shift: Tuple[float, float] = (-0.05, 0.05),
    ):
        self.p = float(p)
        self.h_scale = h_scale
        self.e_scale = e_scale
        self.h_shift = h_shift
        self.e_shift = e_shift

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img = ensure_rgb(img)
        x = np.asarray(img).astype(np.float32) / 255.0

        hs = random.uniform(self.h_scale[0], self.h_scale[1])
        es = random.uniform(self.e_scale[0], self.e_scale[1])
        hh = random.uniform(self.h_shift[0], self.h_shift[1])
        eh = random.uniform(self.e_shift[0], self.e_shift[1])

        x[..., 2] = np.clip(x[..., 2] * hs + hh, 0.0, 1.0)
        x[..., 0] = np.clip(x[..., 0] * es + eh, 0.0, 1.0)
        x[..., 1] = np.clip(x[..., 1] * ((hs + es) / 2.0), 0.0, 1.0)

        y = (x * 255.0).astype(np.uint8)
        return Image.fromarray(y, mode="RGB")


class RCCDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        label2idx: Dict[str, int],
        data_root: str = ".",
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.label2idx = label2idx
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.data_root, p)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = self._resolve_path(str(row[self.path_col]))

        if not os.path.exists(p) and "relpath" in self.df.columns and self.path_col != "relpath":
            p2 = self._resolve_path(str(row["relpath"]))
            if os.path.exists(p2):
                p = p2

        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p} (row {idx})")

        img = ensure_rgb(Image.open(p))
        y = self.label2idx[str(row["label"])]

        if self.transform is not None:
            img = self.transform(img)

        return img, y


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


SUPPORTED_ARCHS = [
    "resnet18",
    "resnet34",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "efficientnet_b0",
    "efficientnet_b2",
    "densenet121",
    "regnety_016",
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
]


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.strip()
    if _HAVE_TIMM:
        if arch not in timm.list_models(pretrained=True) and arch not in timm.list_models(pretrained=False):
            raise ValueError(f"Unknown arch '{arch}' for timm. Try one of: {SUPPORTED_ARCHS}")
        return timm.create_model(arch, pretrained=True, num_classes=num_classes)

    if not _HAVE_TORCHVISION:
        raise RuntimeError("Neither timm nor torchvision is available. Install timm: pip install timm")

    if arch == "resnet18":
        m = resnet18(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "resnet34":
        m = resnet34(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "densenet121":
        m = densenet121(weights="DEFAULT")
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    raise ValueError(f"Arch '{arch}' requires timm. Install timm (recommended): pip install timm")


@torch.no_grad()
def evaluate(model, loader, device, idx2label: Dict[int, str]):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    n = 0

    ce = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = ce(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx2label[i] for i in range(len(idx2label))],
        digits=2,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    cm = confusion_matrix(y_true, y_pred)
    return total_loss / max(1, n), acc, macro_f1, report, cm


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
        n += xb.size(0)
    return running / max(1, n)


def build_transforms(img_size: int, stain: Optional[HEStainJitter] = None):
    if transforms is None:
        raise RuntimeError("torchvision.transforms not available")

    train_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    ]
    if stain is not None:
        train_list.insert(0, stain)

    train_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    val_list = [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    return transforms.Compose(train_list), transforms.Compose(val_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--arch", default="convnext_tiny", choices=SUPPORTED_ARCHS)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data-root", default=".")
    parser.add_argument("--path-col", default=None)

    parser.add_argument("--stain-augment", action="store_true")
    parser.add_argument("--stain-prob", type=float, default=0.5)
    parser.add_argument("--h-scale-min", type=float, default=0.9)
    parser.add_argument("--h-scale-max", type=float, default=1.1)
    parser.add_argument("--e-scale-min", type=float, default=0.9)
    parser.add_argument("--e-scale-max", type=float, default=1.1)
    parser.add_argument("--h-shift-min", type=float, default=-0.05)
    parser.add_argument("--h-shift-max", type=float, default=0.05)
    parser.add_argument("--e-shift-min", type=float, default=-0.05)
    parser.add_argument("--e-shift-max", type=float, default=0.05)

    args = parser.parse_args()
    seed_everything(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    for col in ["label", "patient"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}'. Found: {list(df.columns)}")

    path_col = detect_path_col(df, args.path_col)

    if "split" in df.columns:
        train_df = df[df["split"].astype(str).str.lower().eq("train")].copy()
        val_df = df[df["split"].astype(str).str.lower().eq("val")].copy()
        test_df = df[df["split"].astype(str).str.lower().eq("test")].copy()
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError("CSV has 'split' column but train/val/test rows are missing.")
    else:
        raise ValueError(
            "Your CSV already has split column in your logs; please keep it. "
            "If you removed it, add split=train/val/test at patient-level."
        )

    labels = sorted(df["label"].astype(str).unique().tolist())
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}

    counts = train_df["label"].value_counts().reindex(labels).fillna(0).values.astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv / inv.mean()
    class_weights = torch.tensor(inv, dtype=torch.float32)

    sampler = None
    if args.use_sampler:
        w_per_class = inv
        sample_w = train_df["label"].map({l: w_per_class[label2idx[l]] for l in labels}).values.astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w),
            num_samples=len(sample_w),
            replacement=True,
        )

    stain = None
    if args.stain_augment:
        stain = HEStainJitter(
            p=args.stain_prob,
            h_scale=(args.h_scale_min, args.h_scale_max),
            e_scale=(args.e_scale_min, args.e_scale_max),
            h_shift=(args.h_shift_min, args.h_shift_max),
            e_shift=(args.e_shift_min, args.e_shift_max),
        )

    train_tf, val_tf = build_transforms(args.img_size, stain=stain)

    train_ds = RCCDataset(train_df, path_col=path_col, label2idx=label2idx, data_root=args.data_root, transform=train_tf)
    val_ds = RCCDataset(val_df, path_col=path_col, label2idx=label2idx, data_root=args.data_root, transform=val_tf)
    test_ds = RCCDataset(test_df, path_col=path_col, label2idx=label2idx, data_root=args.data_root, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    model = build_model(args.arch, num_classes=len(labels)).to(device)

    weight = class_weights.to(device) if args.use_class_weights else None
    if args.focal:
        criterion = FocalLoss(gamma=args.gamma, weight=weight, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_f1 = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, device, optimizer, criterion)
        _, _, val_f1, _, _ = evaluate(model, val_loader, device, idx2label)

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = os.path.join(args.outdir, f"{args.model_name}.pth")
    torch.save(model.state_dict(), out_path)


if __name__ == "__main__":
    main()
