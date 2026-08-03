import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ================= #
# DATA REPRESENTATION
# ================= #

MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

MAX_LEN = 300


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_sequence(seq, augment=False):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

    if augment and random.random() > 0.5:
        seq = "".join(complement.get(base, "N") for base in reversed(seq))

    if augment:
        shift = random.randint(-5, 5)
        seq = seq[max(0, shift) :] if shift > 0 else "N" * (-shift) + seq

    seq = seq[:MAX_LEN]
    encoded = [MAPPING.get(base, [0, 0, 0, 0]) for base in seq]

    while len(encoded) < MAX_LEN:
        encoded.append([0, 0, 0, 0])

    return encoded


class DNADataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.ids = dataframe[0].values
        self.sequences = dataframe[1].values
        self.labels = dataframe[2].values
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode_sequence(self.sequences[idx], self.augment),
            dtype=torch.float32,
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ================= #
# MODEL DEFINITION
# ================= #


class MultiKernelCNNStem(nn.Module):
    def __init__(
        self,
        in_channels=4,
        filters_per_kernel=72,
        kernels=(5, 7, 11, 15),
        stride=2,
        dropout=0.25,
    ):
        super().__init__()

        self.kernels = kernels
        self.stride = stride
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    filters_per_kernel,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernels
            ]
        )

        out_channels = filters_per_kernel * len(kernels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = [F.gelu(conv(x)) for conv in self.convs]
        min_length = min(feature.shape[-1] for feature in features)
        features = [feature[..., :min_length] for feature in features]
        x = torch.cat(features, dim=1)
        x = self.dropout(self.norm(x))
        return x.transpose(1, 2)


class ViraExplorerCNNTransformer(nn.Module):
    def __init__(
        self,
        cnn_dropout=0.25,
        transformer_dropout=0.1,
        transformer_dim=288,
        filters_per_kernel=72,
        kernels=(5, 7, 11, 15),
        cnn_stride=2,
        num_layers=6,
        nhead=8,
        max_len=300,
        pooling="cls",
        attention_window=None,
    ):
        super().__init__()

        if pooling not in {"cls", "attention", "cls_attention"}:
            raise ValueError("pooling must be 'cls', 'attention', or 'cls_attention'")

        self.pooling = pooling
        self.attention_window = attention_window

        self.stem = MultiKernelCNNStem(
            in_channels=4,
            filters_per_kernel=filters_per_kernel,
            kernels=kernels,
            stride=cnn_stride,
            dropout=cnn_dropout,
        )

        stem_dim = filters_per_kernel * len(kernels)
        self.projection = (
            nn.Identity()
            if stem_dim == transformer_dim
            else nn.Linear(stem_dim, transformer_dim)
        )

        token_count = min(
            ((max_len + 2 * (kernel // 2) - kernel) // cnn_stride) + 1
            for kernel in kernels
        )
        extra_tokens = 1 if pooling in {"cls", "cls_attention"} else 0

        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)
            if pooling in {"cls", "cls_attention"}
            else None
        )

        self.pos_enc = nn.Parameter(
            torch.randn(1, token_count + extra_tokens, transformer_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.attention_pool = (
            nn.Linear(transformer_dim, 1)
            if pooling in {"attention", "cls_attention"}
            else None
        )

        representation_dim = (
            transformer_dim * 2 if pooling == "cls_attention" else transformer_dim
        )

        self.norm = nn.LayerNorm(representation_dim)

        self.fc = nn.Sequential(
            nn.Linear(representation_dim, 512),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.Linear(128, 1),
        )

    def local_attention_mask(self, size, device, has_cls):
        if self.attention_window is None:
            return None

        positions = torch.arange(size, device=device)
        distance = torch.abs(positions[:, None] - positions[None, :])
        mask = distance > self.attention_window

        if has_cls:
            # Keep CLS global: it can attend to all motif tokens, and all motif
            # tokens can attend back to CLS. The local window applies only among
            # motif tokens.
            mask[0, :] = False
            mask[:, 0] = False

        return mask

    def forward(self, x):
        x = self.projection(self.stem(x))
        batch_size = x.shape[0]

        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_index = 0
        else:
            cls_index = None

        x = x + self.pos_enc[:, : x.shape[1], :]
        mask = self.local_attention_mask(
            x.shape[1],
            x.device,
            has_cls=self.cls_token is not None,
        )
        x = self.transformer(x, mask=mask)

        if self.pooling == "attention":
            weights = torch.softmax(self.attention_pool(x).squeeze(-1), dim=1)
            x = torch.sum(x * weights.unsqueeze(-1), dim=1)
        elif self.pooling == "cls_attention":
            cls_representation = x[:, cls_index, :]
            token_outputs = x[:, 1:, :]
            weights = torch.softmax(
                self.attention_pool(token_outputs).squeeze(-1),
                dim=1,
            )
            pooled_tokens = torch.sum(
                token_outputs * weights.unsqueeze(-1),
                dim=1,
            )
            x = torch.cat([cls_representation, pooled_tokens], dim=1)
        else:
            x = x[:, cls_index, :]

        x = self.norm(x)
        return self.fc(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets.unsqueeze(1)
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            reduction="none",
        )
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


BASE_CONFIG = {
    "cnn_dropout": 0.25,
    "transformer_dropout": 0.1,
    "transformer_dim": 288,
    "filters_per_kernel": 72,
    "kernels": (5, 7, 11, 15),
    "cnn_stride": 2,
    "num_layers": 6,
    "nhead": 8,
    "max_len": 300,
    "pooling": "cls",
    "attention_window": None,
}


VARIANTS = {
    "current-cnn-transformer": {},
    "two-kernels-7-11": {
        "kernels": (7, 11),
        "filters_per_kernel": 144,
        "transformer_dim": 288,
    },
    "four-kernels-128-dim512": {
        "filters_per_kernel": 128,
        "transformer_dim": 512,
    },
    "cls-plus-attention-pooling": {"pooling": "cls_attention"},
    "local-attention-64": {"attention_window": 64},
}


def build_model(config):
    return ViraExplorerCNNTransformer(**config)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast(device_type=device.type, enabled=False)


def predict(model, loader, device):
    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, total=len(loader), desc="Predict", leave=False):
            logits = model(x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probabilities.extend(probs)
            labels.extend(y.numpy())

    return np.array(labels), np.array(probabilities)


def choose_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[int(np.argmax(f1))])


def evaluate_predictions(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "virus_precision": float(tp / (tp + fp + 1e-8)),
        "virus_recall": float(tp / (tp + fn + 1e-8)),
        "virus_f1": float((2 * tp) / (2 * tp + fp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
    }


def save_test_outputs(output_dir, ids, labels, probabilities, threshold, variant_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(labels, probabilities, threshold)

    pd.DataFrame(
        {
            "id": ids,
            "label": labels.astype(int),
            "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, lw=2, label=f"{variant_name} AUROC={metrics['auroc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set(
        title="ROC curve — test set",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        recall,
        precision,
        lw=2,
        color="darkorange",
        label=f"{variant_name} AP={metrics['average_precision']:.3f}",
    )
    axes[1].set(
        title="Precision-recall curve — test set",
        xlabel="Recall",
        ylabel="Precision",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "test_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        labels,
        (probabilities >= threshold).astype(int),
        target_names=["non-virus", "virus"],
        digits=4,
    )
    (output_dir / "classification_report.txt").write_text(report + "\n")

    return metrics


def lr_lambda(epoch, warmup_epochs, epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs

    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_one_variant(
    name,
    config,
    loaders,
    test_ids,
    output_dir,
    device,
    epochs,
    warmup_epochs,
    patience,
    resume,
):
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    config_json = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in config.items()
    }
    (variant_dir / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")

    model = build_model(config).to(device)
    params = parameter_count(model)
    print(f"\n{name}: {params:,} parameters")

    criterion = FocalLoss(alpha=0.75, gamma=2.0, smoothing=0.05)

    transformer_params = list(model.transformer.parameters())
    other_params = [
        parameter
        for param_name, parameter in model.named_parameters()
        if not param_name.startswith("transformer.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": 1e-4},
            {"params": transformer_params, "lr": 5e-5},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: lr_lambda(epoch, warmup_epochs, epochs),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc = -1.0
    best_epoch = 0
    counter = 0
    start_epoch = 0
    history = []

    checkpoint_path = variant_dir / "checkpoint.pt"
    best_model_path = variant_dir / "best_model.pt"
    history_path = variant_dir / "history.csv"

    for parameter in model.transformer.parameters():
        parameter.requires_grad = False

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_auc = checkpoint["best_auc"]
        best_epoch = checkpoint["best_epoch"]
        counter = checkpoint["counter"]
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch >= warmup_epochs:
            for parameter in model.transformer.parameters():
                parameter.requires_grad = True
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed {name} from epoch {start_epoch}")

    started = time.time()

    for epoch in range(start_epoch, epochs):
        if epoch == warmup_epochs:
            for parameter in model.transformer.parameters():
                parameter.requires_grad = True
            print(f"{name}: transformer unfrozen")

        model.train()
        total_loss = 0.0

        for x_batch, y_batch in tqdm(
            loaders["train"],
            total=len(loaders["train"]),
            desc=f"{name} epoch {epoch + 1}/{epochs}",
        ):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device):
                loss = criterion(model(x_batch), y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        val_labels, val_probs = predict(model, loaders["validation"], device)
        val_auc = float(roc_auc_score(val_labels, val_probs))
        train_loss = total_loss / len(loaders["train"])

        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"{name}: saved new best validation AUROC {best_auc:.4f}")
        else:
            counter += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_auroc": val_auc,
            "cnn_lr": optimizer.param_groups[0]["lr"],
            "transformer_lr": optimizer.param_groups[1]["lr"],
            "seconds_elapsed": time.time() - started,
            "best_validation_auroc": best_auc,
            "best_epoch": best_epoch,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_auc": best_auc,
                "best_epoch": best_epoch,
                "counter": counter,
                "config": config_json,
            },
            checkpoint_path,
        )

        print(
            f"{name} epoch {epoch + 1:3d} | "
            f"loss={train_loss:.4f} | val_auroc={val_auc:.4f} | "
            f"best={best_auc:.4f}"
        )

        if patience > 0 and counter >= patience:
            print(f"{name}: early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_labels, val_probs = predict(model, loaders["validation"], device)
    threshold = choose_threshold(val_labels, val_probs)

    test_labels, test_probs = predict(model, loaders["test"], device)
    test_metrics = save_test_outputs(
        variant_dir,
        test_ids,
        test_labels,
        test_probs,
        threshold,
        name,
    )

    result = {
        "variant": name,
        "parameters": params,
        "best_validation_auroc": best_auc,
        "best_epoch": best_epoch,
        "training_minutes": (time.time() - started) / 60,
        **test_metrics,
    }
    (variant_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def save_comparison_outputs(results, output_dir):
    ranked = sorted(results, key=lambda row: row["best_validation_auroc"], reverse=True)
    pd.DataFrame(ranked).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(ranked, indent=2) + "\n")

    names = [row["variant"] for row in ranked]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].barh(
        names[::-1],
        [row["best_validation_auroc"] for row in ranked][::-1],
    )
    axes[0, 0].set(title="Best validation AUROC", xlim=(0.5, 1.0))

    axes[0, 1].barh(names[::-1], [row["auroc"] for row in ranked][::-1])
    axes[0, 1].set(title="Test AUROC of validation-selected checkpoint", xlim=(0.5, 1.0))

    axes[1, 0].barh(names[::-1], [row["training_minutes"] for row in ranked][::-1])
    axes[1, 0].set(title="Training time", xlabel="Minutes")

    axes[1, 1].scatter(
        [row["parameters"] / 1_000_000 for row in ranked],
        [row["best_validation_auroc"] for row in ranked],
        s=80,
    )
    for row in ranked:
        axes[1, 1].annotate(
            row["variant"],
            (row["parameters"] / 1_000_000, row["best_validation_auroc"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1, 1].set(
        title="Capacity vs validation AUROC",
        xlabel="Parameters (millions)",
        ylabel="Validation AUROC",
        ylim=(0.5, 1.0),
    )

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    fig.suptitle("ViraExplorer CNN–Transformer trial 2 comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "variant_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 7))
    for row in ranked:
        history_path = output_dir / row["variant"] / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            axis.plot(
                history["epoch"],
                history["validation_auroc"],
                marker="o",
                markersize=2.5,
                label=row["variant"],
            )
    axis.set(
        title="Validation AUROC learning curves",
        xlabel="Epoch",
        ylabel="Validation AUROC",
        ylim=(0.5, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_learning_curves.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare trial-2 viraminer3 CNN-Transformer variants"
    )
    parser.add_argument("--data-path", default="/home/fusaro/viraminer/dataset")
    parser.add_argument(
        "--save-dir",
        default="/home/fusaro/viraminer/cnn_transformer_variants_trial2",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience. Set 0 to disable early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help="Comma-separated variant names to run",
    )
    return parser.parse_args()


def select_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    train = pd.read_csv(data_path / "fullset_train.csv", header=None)
    validation = pd.read_csv(data_path / "fullset_validation.csv", header=None)
    test = pd.read_csv(data_path / "fullset_test.csv", header=None)

    print("Train shape:", train.shape)
    print("Sequence length:", len(train[1][0]))
    print("\nClass balance:")
    print(train[2].value_counts())

    loaders = {
        "train": DataLoader(
            DNADataset(train, augment=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "validation": DataLoader(
            DNADataset(validation),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            DNADataset(test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    device = select_device(args.device)
    print(f"Device: {device}")

    selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected_variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {list(VARIANTS)}")

    results = []
    for name in selected_variants:
        config = {**BASE_CONFIG, **VARIANTS[name]}
        result = train_one_variant(
            name=name,
            config=config,
            loaders=loaders,
            test_ids=test[0].values,
            output_dir=output_dir,
            device=device,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            resume=args.resume,
        )
        results.append(result)
        save_comparison_outputs(results, output_dir)

    save_comparison_outputs(results, output_dir)
    winner = max(results, key=lambda row: row["best_validation_auroc"])
    print(
        "\nBest variant by validation AUROC: "
        f"{winner['variant']} ({winner['best_validation_auroc']:.4f})"
    )
    print(f"Winner test AUROC: {winner['auroc']:.4f}")
    print(f"Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
