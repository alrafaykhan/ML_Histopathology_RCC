#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import random
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as tvm
import timm

from sklearn.metrics import classification_report, accuracy_score, f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str, num_classes: int) -> Tuple[nn.Module, int]:
    arch = arch.lower()
    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, 224
    if arch == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, 224
    if arch == "convnext_tiny":
        m = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes)
        return m, 224
    if arch == "efficientnet_b0":
        m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
        return m, 224
    if arch == "efficientnet_b2":
        m = timm.create_model("efficientnet_b2", pretrained=True, num_classes=num_classes)
        return m, 260
    raise ValueError(f"Unknown arch: {arch}")


class ImageCSV(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, img_size: int, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.img_size = img_size

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        label = row["label"]
        y = self.class_to_idx[label]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.tf(im)
        return x, y


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
        )
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(labels: List[int], num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    w = total / (num_classes * np.maximum(counts, 1.0))
    return w


def patient_level_split(df: pd.DataFrame, val_size=0.25, test_size=0.25, seed=42):
    if "patient" not in df.columns:
        df["patient"] = df["filepath"].apply(lambda p: Path(p).parent.name)

    patients = df["patient"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    test_pat = set(patients[:n_test])
    val_pat = set(patients[n_test:n_test + n_val])
    train_pat = set(patients[n_test + n_val:])

    tr = df[df["patient"].isin(train_pat)].copy()
    va = df[df["patient"].isin(val_pat)].copy()
    te = df[df["patient"].isin(test_pat)].copy()
    return tr, va, te


def make_dataloaders(tr_df, va_df, te_df, class_to_idx, img_size, bs, use_class_weights):
    num_classes = len(class_to_idx)

    train_ds = ImageCSV(tr_df, class_to_idx, img_size, is_train=True)
    val_ds = ImageCSV(va_df, class_to_idx, img_size, is_train=False)
    test_ds = ImageCSV(te_df, class_to_idx, img_size, is_train=False)

    if use_class_weights:
        train_labels_idx = [class_to_idx[lbl] for lbl in tr_df["label"].tolist()]
        class_w = compute_class_weights(train_labels_idx, num_classes)
        sample_w = class_w[train_labels_idx]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_w),
            num_samples=len(sample_w),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True
        )

    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def run_eval(model, loader, device, idx_to_class):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.cpu().numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with columns: filepath,label[,patient]")
    p.add_argument(
        "--arch",
        choices=["resnet18", "resnet34", "convnext_tiny", "efficientnet_b0", "efficientnet_b2"],
        default="resnet18",
    )
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--focal", action="store_true")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--use-class-weights", action="store_true")
    p.add_argument("--val-size", type=float, default=0.25)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="models")
    p.add_argument("--model-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "filepath" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have at least columns: 'filepath','label' (and optionally 'patient').")
    df["filepath"] = df["filepath"].astype(str)

    classes = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    tr_df, va_df, te_df = patient_level_split(
        df, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        tr_df, va_df, te_df, class_to_idx, args.img_size, args.bs, args.use_class_weights
    )

    model, _ = build_model(args.arch, num_classes=len(classes))
    device = torch.device(args.device)
    model.to(device)

    if args.use_class_weights:
        train_labels_idx = [class_to_idx[lbl] for lbl in tr_df["label"].tolist()]
        class_w = compute_class_weights(train_labels_idx, len(classes))
        torch_class_w = torch.tensor(class_w, dtype=torch.float32, device=device)
    else:
        class_w = None
        torch_class_w = None

    if args.focal:
        loss_fn = FocalLoss(gamma=args.gamma, alpha=class_w if class_w is not None else None)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch_class_w, label_smoothing=args.label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.model_name is None:
        args.model_name = f"{args.arch}_rcc_baseline"
    save_path = os.path.join(args.outdir, f"{args.model_name}.pth")

    best_val_macro_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        val_acc, val_macro_f1, _ = run_eval(model, val_loader, device, idx_to_class)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stop and epochs_no_improve >= args.early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    torch.save(model.state_dict(), save_path)

    _ = run_eval(model, test_loader, device, idx_to_class)


if __name__ == "__main__":
    main()
