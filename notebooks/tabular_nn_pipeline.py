"""
End-to-end neural network pipeline for the AITH DL Competition – Tabular data (2025).

Key guarantees:
- Uses only a PyTorch MLP (no tree models).
- Per-fold preprocessing (imputation + scaling) fit only on the training split.
- StratifiedKFold CV with OOF ROC AUC reporting.
- Early stopping based solely on validation ROC AUC.
- Deterministic seeds for reproducibility.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


# --------------------
# Reproducibility setup
# --------------------
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------
# Feature handling
# --------------------
TARGET_COL = "smoking"
ID_COLUMNS: List[str] = ["id"]  # kept for ordering only; not used as a feature


def detect_feature_types(df: pd.DataFrame, target_col: str, id_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Identify categorical and numerical columns."""
    feature_cols = [c for c in df.columns if c not in set(id_cols) | {target_col}]
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype == "object" or str(dtype).startswith("category"):
            cat_cols.append(col)
        elif pd.api.types.is_integer_dtype(dtype) and df[col].nunique() < 20:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols


class FoldPreprocessor:
    """Per-fold preprocessing: median imputation + scaling for numeric; ordinal encoding for categoricals."""

    def __init__(self, cat_cols: Sequence[str], num_cols: Sequence[str]) -> None:
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.num_impute_: Optional[pd.Series] = None
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) if self.cat_cols else None

    def fit(self, df: pd.DataFrame) -> "FoldPreprocessor":
        if self.num_cols:
            num_df = df[self.num_cols]
            self.num_impute_ = num_df.median()
            num_filled = num_df.fillna(self.num_impute_)
            self.scaler.fit(num_filled)
        if self.cat_cols and self.encoder is not None:
            cat_df = df[self.cat_cols].astype("object").fillna("missing")
            self.encoder.fit(cat_df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []
        if self.num_cols:
            assert self.num_impute_ is not None, "Numeric imputer not fitted."
            num_df = df[self.num_cols].fillna(self.num_impute_)
            parts.append(self.scaler.transform(num_df))
        if self.cat_cols and self.encoder is not None:
            cat_df = df[self.cat_cols].astype("object").fillna("missing")
            parts.append(self.encoder.transform(cat_df))
        if not parts:
            raise ValueError("No features available after preprocessing.")
        return np.hstack(parts).astype(np.float32)


# --------------------
# Dataset and model
# --------------------
class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None) -> None:
        self.features = torch.from_numpy(features).float()
        self.targets = None if targets is None else torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.targets is None:
            return self.features[idx]
        return self.features[idx], self.targets[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------
# Training utilities
# --------------------
def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> Tuple[dict, float]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds: List[np.ndarray] = []
        val_targets: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(yb.numpy())

        val_pred = np.concatenate(val_preds)
        val_y = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_y, val_pred)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_state, best_auc


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds)


# --------------------
# Pipeline
# --------------------
def run_cv_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = train_df[TARGET_COL].values.astype(np.float32)
    cat_cols, num_cols = detect_feature_types(train_df, TARGET_COL, ID_COLUMNS)

    print(f"Detected {len(num_cols)} numerical and {len(cat_cols)} categorical features.")
    if not num_cols and not cat_cols:
        raise ValueError("No features found for training.")

    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    fold_aucs: List[float] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y), start=1):
        print(f"\nFold {fold}/{n_splits}")
        tr_df = train_df.iloc[train_idx].reset_index(drop=True)
        va_df = train_df.iloc[val_idx].reset_index(drop=True)

        preprocessor = FoldPreprocessor(cat_cols=cat_cols, num_cols=num_cols)
        preprocessor.fit(tr_df)

        X_tr = preprocessor.transform(tr_df)
        X_va = preprocessor.transform(va_df)
        X_te = preprocessor.transform(test_df)

        y_tr = tr_df[TARGET_COL].values.astype(np.float32)
        y_va = va_df[TARGET_COL].values.astype(np.float32)

        train_ds = TabularDataset(X_tr, y_tr)
        val_ds = TabularDataset(X_va, y_va)
        test_ds = TabularDataset(X_te)

        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False)

        model = MLP(input_dim=X_tr.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)

        best_state, best_auc = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
        )
        fold_aucs.append(best_auc)
        print(f"Fold {fold} best ROC AUC: {best_auc:.5f}")

        model.load_state_dict(best_state)

        oof_pred[val_idx] = predict_proba(model, val_loader, device)
        test_pred += predict_proba(model, test_loader, device) / n_splits

    cv_auc = roc_auc_score(y, oof_pred)
    print(f"\nOOF ROC AUC: {cv_auc:.5f}")
    return oof_pred, test_pred, fold_aucs


def main() -> None:
    set_seed()

    data_dir = Path("data/competition")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_sub_path = data_dir / "sample_submission.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    # Align test rows to sample submission order and drop ID column from features.
    test_df = test_df.merge(sample_sub[["id"]], on="id", how="right")
    feature_test_df = test_df.drop(columns=ID_COLUMNS)
    feature_train_df = train_df.drop(columns=ID_COLUMNS)

    oof_pred, test_pred, fold_aucs = run_cv_pipeline(
        train_df=feature_train_df,
        test_df=feature_test_df,
        n_splits=5,
        hidden_dims=(256, 128, 64),
        dropout=0.2,
        batch_size=256,
        max_epochs=100,
        patience=10,
        lr=1e-3,
        weight_decay=1e-4,
    )

    submission = sample_sub.copy()
    submission["smoking"] = np.clip(test_pred, 0.0, 1.0)
    output_path = Path("submission.csv")
    submission.to_csv(output_path, index=False)

    print("\nFold ROC AUCs:", [f"{auc:.5f}" for auc in fold_aucs])
    print(f"Mean ROC AUC: {np.mean(fold_aucs):.5f}")
    print(f"Submission saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
