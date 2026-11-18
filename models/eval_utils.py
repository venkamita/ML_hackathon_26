from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score

from features.build_candidates import FEATURE_COLS

RANDOM_STATE = 42


def load_usable(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a feature parquet, keep usable rows, and apply the required ordering."""
    data = pd.read_parquet(path)
    usable = data[data["label_usable"] == 1].copy()
    usable = usable.sort_values(["trip_id", "point_idx", "cand_rank"], kind="mergesort")
    return data, usable


def trip_folds(trip_ids: pd.Series | list[str], n_splits: int = 5) -> dict[str, int]:
    sorted_ids = sorted(set(trip_ids))
    return {trip_id: position % n_splits for position, trip_id in enumerate(sorted_ids)}


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / max(float(temperature), 1e-8)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.maximum(scores, 0.0)
    total = scores.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(scores), 1.0 / len(scores))
    return scores / total


def per_point_probabilities(df: pd.DataFrame, scores: np.ndarray, temperature: float = 1.0,
                           real_valued: bool = False) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    probabilities = np.empty(len(df), dtype=float)
    if len(scores) != len(df):
        raise ValueError("scores and dataframe must have equal lengths")
    for positions in df.groupby(["trip_id", "point_idx"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        if real_valued:
            probabilities[positions] = softmax(scores[positions], temperature)
        else:
            probabilities[positions] = normalise_scores(scores[positions])
    return probabilities


def top1_predictions(df: pd.DataFrame, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return chosen-row indices and row-level top-1 predictions."""
    pred_rows = np.zeros(len(df), dtype=np.int8)
    chosen_indices = []
    start = 0
    for positions in df.groupby(["trip_id", "point_idx"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        local = probabilities[positions]
        ranks = df.iloc[positions]["cand_rank"].to_numpy()
        order = np.lexsort((ranks, -local))
        chosen = positions[order[0]]
        chosen_indices.append(chosen)
        pred_rows[chosen] = 1
    return np.asarray(chosen_indices, dtype=int), pred_rows


def calculate_metrics(df: pd.DataFrame, probabilities: np.ndarray) -> dict:
    y = df["label"].to_numpy(dtype=int)
    chosen, pred = top1_predictions(df, probabilities)
    top1_correct = 0
    point_count = 0
    for positions in df.groupby(["trip_id", "point_idx"], sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        local_prob = probabilities[positions]
        ranks = df.iloc[positions]["cand_rank"].to_numpy()
        winner = positions[np.lexsort((ranks, -local_prob))[0]]
        if df.iloc[winner]["label"] == 1:
            top1_correct += 1
        point_count += 1

    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "top1_acc": top1_correct / point_count,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_1": float(precision_score(y, pred, average="binary", zero_division=0)),
        "recall_1": float(recall_score(y, pred, average="binary", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7))),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def mean_std_metrics(rows: list[dict], metric_names: list[str]) -> dict:
    result = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_std"] = float(values.std(ddof=0))
    return result


def save_confusion_matrix(cm: list[list[int]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(np.asarray(cm), cmap="Blues")
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_learning_curves(evals_result: dict, path: Path, title: str) -> None:
    if not evals_result:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset_name, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            ax.plot(values, label=f"{dataset_name}-{metric_name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_oof(train: pd.DataFrame, probabilities: np.ndarray, fold_assignment: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "trip_id": train["trip_id"].to_numpy(),
        "point_idx": train["point_idx"].to_numpy(),
        "arc_id": train["arc_id"].to_numpy(),
        "cand_rank": train["cand_rank"].to_numpy(),
        "label": train["label"].to_numpy(),
        "fold": fold_assignment,
        "prob": probabilities,
    })


def inner_validation_trips(train_fold_trips: list[str], seed: int = RANDOM_STATE) -> tuple[list[str], list[str]]:
    """Deterministically hold out 20% of trips from the outer training trips."""
    rng = np.random.default_rng(seed)
    trips = np.asarray(sorted(train_fold_trips))
    perm = rng.permutation(len(trips))
    n_inner = max(1, int(np.ceil(0.20 * len(trips))))
    val = sorted(trips[perm[:n_inner]].tolist())
    fit = sorted(trips[perm[n_inner:]].tolist())
    return fit, val
