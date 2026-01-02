#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TCGA_PAT = re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', re.I)
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def get_patient_id(name: str) -> str:
    m = TCGA_PAT.search(name)
    if m:
        return m.group(1)
    base = Path(name).name.split('.')[0]
    parts = base.split('-')
    if len(parts) >= 3 and parts[0].upper() == 'TCGA':
        return '-'.join(parts[:3])
    return base

def main(root, out_csv, seed, train, val, test):
    root = Path(root)
    rows = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(root)
            top = rel.parts[0]
            label = 'normal' if any('normal' in part.lower() for part in rel.parts) else top
            patient_id = get_patient_id(p.name)
            rows.append({
                'path': str(p.resolve()),
                'relpath': str(rel),
                'label': label,
                'patient_id': patient_id
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No images found.')

    df['label'] = df['label'].str.strip()
    classes = sorted(df['label'].unique().tolist())
    df = df[df['label'].isin(classes)].reset_index(drop=True)

    assert abs(train + val + test - 1.0) < 1e-6
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val + test), random_state=seed)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(test / (val + test)), random_state=seed)

    groups = df['patient_id']
    idx = df.index.values
    train_idx, temp_idx = next(gss1.split(idx, df['label'], groups))
    temp_df = df.loc[temp_idx]
    val_idx, test_idx = next(
        gss2.split(temp_df.index.values, temp_df['label'], temp_df['patient_id'])
    )

    df['split'] = 'train'
    df.loc[temp_df.index[val_idx], 'split'] = 'val'
    df.loc[temp_df.index[test_idx], 'split'] = 'test'

    df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out_csv', default='rcc_manifest.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train', type=float, default=0.7)
    ap.add_argument('--val', type=float, default=0.15)
    ap.add_argument('--test', type=float, default=0.15)
    args = ap.parse_args()
    main(args.root, args.out_csv, args.seed, args.train, args.val, args.test)
