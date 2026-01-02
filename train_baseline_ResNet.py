#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a stronger RCC baseline (ResNet-18 by default) from a CSV manifest.

CSV expected columns (case-insensitive):
- path: absolute or relative path to image file
- label: class name (e.g., ccRCC, pRCC, chRCC, normal)
- patient or patient_id (optional): if absent, we try to infer a TCGA-like barcode from filename

Example:
python train_baseline.py --csv rcc_manifest_clean.csv --epochs 20 --bs 32 --focal --early-stop 7
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools
import warnings

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def infer_patient_from_path(p: str) -> str:
    stem = Path(p).stem
    parts = stem.split('-')
    if len(parts) >= 3 and all(len(x) > 0 for x in parts[:3]):
        return "-".join(parts[:3])
    return Path(p).parent.name


def norm_classname(c: str) -> str:
    return c.strip().replace(" ", "").replace("\t", "").replace("\n", "")


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(list(set(labels)))
    str2idx = {s: i for i, s in enumerate(uniq)}
    idx2str = {i: s for s, i in str2idx.items()}
    return str2idx, idx2str


def color_distortion_jitter(strength=0.4):
    return transforms.ColorJitter(
        brightness=0.2 * strength,
        contrast=0.2 * strength,
        saturation=0.2 * strength
    )


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', filename='confusion_matrix.png'):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() * 0.6
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=10
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, bbox_inches='tight', dpi=160)
    plt.close()


class RCCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, str2idx: Dict[str, int], train: bool = True, img_size: int = 224):
        self.df = df.reset_index(drop=True).copy()
        self.str2idx = str2idx
        self.train = train

        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomApply([color_distortion_jitter(1.0)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = row['path']
        label = self.str2idx[norm_classname(str(row['label']))]

        with Image.open(path) as img:
            img = img.convert('RGB')
            img = self.tf(img)

        return img, label


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        pt = probs[torch.arange(logits.size(0), device=logits.device), targets]
        ce = F.nll_loss(log_probs, targets, reduction='none', weight=self.alpha)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, idx2str):
    model.eval()
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    preds = all_logits.argmax(dim=1).numpy()
    targets = all_targets.numpy()

    labels_order = list(range(len(idx2str)))
    target_names = [idx2str[i] for i in labels_order]
    report = classification_report(
        targets, preds,
        labels=labels_order,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    macro_f1 = f1_score(targets, preds, labels=labels_order, average='macro', zero_division=0)
    cm = confusion_matrix(targets, preds, labels=labels_order)

    return report, macro_f1, cm, preds, targets


def build_model(num_classes: int, arch: str = 'resnet18', pretrained: bool = True):
    if arch.lower() == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feats, num_classes)
        )
        return m
    if arch.lower() == 'resnet34':
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feats, num_classes)
        )
        return m
    raise ValueError(f"Unsupported arch: {arch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True, help='CSV with columns: path,label[,patient]')
    ap.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--label-smoothing', type=float, default=0.05)
    ap.add_argument('--focal', action='store_true', help='Use focal loss (overrides label smoothing CE)')
    ap.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    ap.add_argument('--use-class-weights', action='store_true', help='Use class weights in loss/sampler')
    ap.add_argument('--val-size', type=float, default=0.2, help='Validation size fraction')
    ap.add_argument('--test-size', type=float, default=0.2, help='Test size fraction (from remaining)')
    ap.add_argument('--early-stop', type=int, default=0, help='Patience on macro-F1 (0=disabled)')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--outdir', type=str, default='models')
    ap.add_argument('--model-name', type=str, default=None, help='Override saved model filename')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device)
    pin_memory = (device.type == 'cuda')

    df = pd.read_csv(args.csv)
    df.columns = [c.lower().strip() for c in df.columns]
    assert 'path' in df.columns and 'label' in df.columns, "CSV must have 'path' and 'label' columns."

    df['path'] = df['path'].astype(str)
    df['label'] = df['label'].astype(str).map(norm_classname)

    patient_col = 'patient' if 'patient' in df.columns else ('patient_id' if 'patient_id' in df.columns else None)
    if patient_col is None:
        df['patient'] = df['path'].apply(infer_patient_from_path)
        patient_col = 'patient'
    else:
        df['patient'] = df[patient_col].astype(str)

    exists_mask = df['path'].apply(lambda p: Path(p).exists())
    df = df[exists_mask].reset_index(drop=True)

    str2idx, idx2str = build_label_maps(df['label'].tolist())
    df['y'] = df['label'].map(str2idx)

    groups = df['patient'].values
    y_all = df['y'].values
    X_index = np.arange(len(df))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=SEED)
    trainrest_idx, val_idx = next(gss1.split(X_index, y_all, groups=groups))

    rest_frac = args.test_size / (1.0 - args.val_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rest_frac, random_state=SEED)
    train_idx, test_idx = next(gss2.split(trainrest_idx, y_all[trainrest_idx], groups=groups[trainrest_idx]))
    train_idx = trainrest_idx[train_idx]
    test_idx = trainrest_idx[test_idx]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    train_ds = RCCDataset(df_train, str2idx, train=True, img_size=args.img_size)
    val_ds = RCCDataset(df_val, str2idx, train=False, img_size=args.img_size)
    test_ds = RCCDataset(df_test, str2idx, train=False, img_size=args.img_size)

    if args.use_class_weights:
        counts = df_train['y'].value_counts().sort_index().values.astype(np.float32)
        class_weights = 1.0 / (counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        sw = class_weights[df_train['y'].values]
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sw), num_samples=len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.bs, sampler=sampler, num_workers=4, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=pin_memory)

    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=pin_memory)

    num_classes = len(str2idx)
    model = build_model(num_classes, arch=args.arch, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.use_class_weights:
        counts = df_train['y'].value_counts().sort_index().values.astype(np.float32)
        cw = 1.0 / (counts + 1e-6)
        cw = cw / cw.sum() * len(cw)
        class_weight_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        class_weight_tensor = None

    if args.focal:
        criterion = FocalLoss(alpha=class_weight_tensor, gamma=args.gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=args.label_smoothing)

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience = args.early_stop
    no_improv = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        _, macro_f1, cm, _, _ = evaluate(model, val_loader, device, idx2str)

        plot_confusion_matrix(
            cm,
            classes=[idx2str[i] for i in range(num_classes)],
            normalize=False,
            title=f'Val Confusion (epoch {epoch})',
            filename=f'{args.outdir}/val_cm_epoch{epoch}.png'
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improv = 0
        else:
            no_improv += 1

        if patience > 0 and no_improv >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _, _, cm, _, _ = evaluate(model, val_loader, device, idx2str)
    plot_confusion_matrix(
        cm,
        classes=[idx2str[i] for i in range(num_classes)],
        normalize=True,
        title='Val Confusion (best, norm)',
        filename=f'{args.outdir}/val_cm_best_norm.png'
    )

    model_name = args.model_name or f"{args.arch}_rcc_baseline.pth"
    save_path = os.path.join(args.outdir, model_name)
    torch.save({
        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'arch': args.arch,
        'label_map': {k: int(v) for k, v in str2idx.items()},
        'idx2str': {int(k): v for k, v in idx2str.items()},
        'img_size': args.img_size,
    }, save_path)

    _, _, cm, _, _ = evaluate(model, test_loader, device, idx2str)
    plot_confusion_matrix(
        cm,
        classes=[idx2str[i] for i in range(num_classes)],
        normalize=True,
        title='Test Confusion (norm)',
        filename=f'{args.outdir}/test_cm_norm.png'
    )


if __name__ == '__main__':
    main()
