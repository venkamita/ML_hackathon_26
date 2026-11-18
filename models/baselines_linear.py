"""
PathSnap: Baseline and Linear Models (Person C - Shanteri).
Implements and evaluates:
  (0) nearest_arc: Classic distance-based Gaussian emission baseline.
  (1) conditional_logit: Point-grouped linear utility competition with analytic gradient.
  (2) gaussian_nb: Log1p-transformed Gaussian Naive Bayes.

Strictly adheres to COMMON RULES:
  - Rows filtered by label_usable == 1, sorted by (trip_id, point_idx, cand_rank)
  - Trip-grouped 5-fold cross-validation: fold = (position in sorted trip list) % 5
  - Transforms and scalers fitted on training folds only
  - Probabilities normalised per GPS point to sum to 1
  - Consistent metric evaluation: top1_acc, macro_f1, precision_1, recall_1, log_loss
  - Standard outputs: cv_folds.csv, summary.json, oof_train.parquet, pred_student_test.parquet,
    confusion_matrix.png, model.joblib
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.build_candidates import FEATURE_COLS

RANDOM_STATE = 42
SIGMA_EMISSION = 6.68
LOG1P_COLS = [
    "dist_m", "dist_minus_min_m", "arc_length_m", "step_next_m",
    "d_prev2_m", "d_prev1_m", "d_next1_m", "d_next2_m", "d_nbr_mean_m"
]


# ==============================================================================
# Helper Functions: Data Prep, Groupings, and Metrics
# ==============================================================================

def load_and_preprocess_data(repo_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads features_train.parquet and features_student_test.parquet,
    filters by label_usable == 1, sorts by (trip_id, point_idx, cand_rank),
    and assigns the deterministic trip-grouped 5 folds.
    """
    train_path = repo_root / "outputs" / "features" / "features_train.parquet"
    test_path = repo_root / "outputs" / "features" / "features_student_test.parquet"

    print(f"Loading data from {train_path.name} and {test_path.name}...")
    df_train_raw = pd.read_parquet(train_path)
    df_test_raw = pd.read_parquet(test_path)

    # Filter label_usable == 1 and sort
    df_train = (
        df_train_raw[df_train_raw["label_usable"] == 1]
        .sort_values(["trip_id", "point_idx", "cand_rank"])
        .reset_index(drop=True)
    )
    df_test = (
        df_test_raw[df_test_raw["label_usable"] == 1]
        .sort_values(["trip_id", "point_idx", "cand_rank"])
        .reset_index(drop=True)
    )

    # Assign 5 folds by sorted unique trip_ids
    sorted_trips = sorted(df_train["trip_id"].unique())
    trip_to_fold = {tid: idx % 5 for idx, tid in enumerate(sorted_trips)}
    df_train["fold"] = df_train["trip_id"].map(trip_to_fold).astype(int)

    print(f"Train: {len(df_train):,} usable rows across {len(sorted_trips)} trips.")
    print(f"Student Test: {len(df_test):,} usable rows across {df_test['trip_id'].nunique()} trips.")
    return df_train, df_test


def get_point_group_info(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a DataFrame sorted by (trip_id, point_idx, cand_rank),
    returns point_ids (0..G-1), start index of each point, and count of candidates per point.
    """
    point_ids = df.groupby(["trip_id", "point_idx"], sort=False).ngroup().values
    point_starts = np.r_[0, np.flatnonzero(point_ids[1:] != point_ids[:-1]) + 1]
    point_lens = np.diff(np.r_[point_starts, len(df)])
    return point_ids, point_starts, point_lens


def normalize_probabilities_per_point(
    scores: np.ndarray,
    point_starts: np.ndarray,
    point_lens: np.ndarray,
    is_log_scale: bool = False
) -> np.ndarray:
    """
    Normalizes candidate scores so that sum(p) == 1 for each GPS observation point.
    If is_log_scale=True, computes numerically stable softmax.
    If is_log_scale=False, computes p = s / sum(s) (with uniform fallback if sum==0).
    """
    if is_log_scale:
        u_max = np.maximum.reduceat(scores, point_starts)
        u_shift = scores - np.repeat(u_max, point_lens)
        exp_u = np.exp(u_shift)
        sum_exp = np.add.reduceat(exp_u, point_starts)
        probs = exp_u / np.repeat(sum_exp, point_lens)
        return probs
    else:
        # Non-negative scores
        sums = np.add.reduceat(scores, point_starts)
        row_sums = np.repeat(sums, point_lens)
        zero_mask = (row_sums == 0)
        # Uniform fallback for zero-sum points
        safe_sums = np.where(zero_mask, 1.0, row_sums)
        probs = scores / safe_sums
        if np.any(zero_mask):
            uniform_p = 1.0 / np.repeat(point_lens, point_lens)
            probs[zero_mask] = uniform_p[zero_mask]
        return probs


def evaluate_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    infer_time_s: float = 0.0,
    train_time_s: float = 0.0
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Computes all Common Rules metrics:
      - top1_acc: highest p (ties to lower cand_rank); correct if label == 1.
      - row-level prediction (top-1 row = 1, others = 0):
          macro_f1, precision_1, recall_1, confusion_matrix
      - log_loss on row-level clipped probabilities
      - infer_ms_per_1k_rows, train_time_s
    """
    df_eval = df[["trip_id", "point_idx", "cand_rank", "label"]].copy()
    df_eval["prob"] = probs

    # Top-1 selection: sort by (trip_id, point_idx, prob DESC, cand_rank ASC)
    df_sorted = df_eval.sort_values(
        ["trip_id", "point_idx", "prob", "cand_rank"],
        ascending=[True, True, False, True]
    )
    top1_indices = df_sorted.groupby(["trip_id", "point_idx"], sort=False).head(1).index.values

    pred = np.zeros(len(df_eval), dtype=int)
    pred[top1_indices] = 1

    y_true = df_eval["label"].values
    num_points = len(top1_indices)
    top1_acc = float(y_true[top1_indices].sum() / num_points)

    macro_f1 = float(f1_score(y_true, pred, average="macro"))
    precision_1 = float(precision_score(y_true, pred, pos_label=1, zero_division=0))
    recall_1 = float(recall_score(y_true, pred, pos_label=1, zero_division=0))

    clipped_probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    loss = float(log_loss(y_true, clipped_probs))

    n_rows = len(df_eval)
    infer_ms_per_1k = float((infer_time_s / n_rows) * 1000.0 * 1000.0) if n_rows > 0 else 0.0

    cm = confusion_matrix(y_true, pred)

    metrics = {
        "top1_acc": round(top1_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "precision_1": round(precision_1, 6),
        "recall_1": round(recall_1, 6),
        "log_loss": round(loss, 6),
        "train_time_s": round(train_time_s, 4),
        "infer_ms_per_1k_rows": round(infer_ms_per_1k, 4),
        "n_rows": n_rows,
        "n_points": num_points
    }
    return metrics, pred, cm


def plot_and_save_confusion_matrix(cm: np.ndarray, out_path: Path, model_name: str):
    """Generates and saves a clean confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Negative (0)", "Positive (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Confusion Matrix: {model_name}"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ==============================================================================
# Model Implementations
# ==============================================================================

class NearestArcModel:
    """Model (0): Nearest Arc Gaussian Emission Baseline."""
    def __init__(self, sigma: float = SIGMA_EMISSION):
        self.sigma = sigma

    def predict_probs(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        t0 = time.time()
        _, pt_starts, pt_lens = get_point_group_info(df)
        dists = df["dist_m"].values
        # Gaussian GPS noise emission score
        raw_scores = np.exp(-0.5 * (dists / self.sigma) ** 2)
        probs = normalize_probabilities_per_point(raw_scores, pt_starts, pt_lens, is_log_scale=False)
        infer_time = time.time() - t0
        return probs, infer_time


class Log1pGaussianNBModel:
    """Model (2): Log1p-transformed Gaussian Naive Bayes."""
    def __init__(self):
        self.nb = GaussianNB()
        self.log_cols = LOG1P_COLS
        self.feature_cols = FEATURE_COLS

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy().values
        for col_name in self.log_cols:
            col_idx = self.feature_cols.index(col_name)
            X[:, col_idx] = np.log1p(np.maximum(X[:, col_idx], 0.0))
        return X

    def fit(self, df_train: pd.DataFrame) -> float:
        t0 = time.time()
        X_tr = self._transform(df_train)
        y_tr = df_train["label"].values
        self.nb.fit(X_tr, y_tr)
        return time.time() - t0

    def predict_probs(self, df_val: pd.DataFrame) -> Tuple[np.ndarray, float]:
        t0 = time.time()
        X_val = self._transform(df_val)
        _, pt_starts, pt_lens = get_point_group_info(df_val)
        raw_scores = self.nb.predict_proba(X_val)[:, 1]
        probs = normalize_probabilities_per_point(raw_scores, pt_starts, pt_lens, is_log_scale=False)
        infer_time = time.time() - t0
        return probs, infer_time


class ConditionalLogitModel:
    """Model (1): Point-Grouped Conditional Logit Model."""
    def __init__(self, lmb: float = 1e-3, max_iter: int = 100):
        self.lmb = lmb
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self.beta: Optional[np.ndarray] = None

    def fit(self, df_train: pd.DataFrame) -> float:
        t0 = time.time()
        X_raw = df_train[self.feature_cols].values
        y = df_train["label"].values
        X = self.scaler.fit_transform(X_raw)

        # Identify point groups
        pt_ids, pt_starts, pt_lens = get_point_group_info(df_train)

        # Filter points that have >= 1 positive
        pt_pos_sums = np.add.reduceat(y, pt_starts)
        pt_has_pos = pt_pos_sums > 0
        G = int(np.sum(pt_has_pos))

        # Restrict training rows to points with >= 1 positive
        valid_row_mask = np.repeat(pt_has_pos, pt_lens)
        X_v = X[valid_row_mask]
        y_v = y[valid_row_mask]
        pt_ids_v = pt_ids[valid_row_mask]

        _, pt_starts_v = np.unique(pt_ids_v, return_index=True)
        pt_starts_v = np.sort(pt_starts_v)
        pt_lens_v = np.diff(np.r_[pt_starts_v, len(y_v)])

        lmb = self.lmb

        def loss_and_grad(beta):
            u_score = X_v @ beta
            u_max = np.maximum.reduceat(u_score, pt_starts_v)
            u_shift = u_score - np.repeat(u_max, pt_lens_v)
            exp_u = np.exp(u_shift)
            sum_exp = np.add.reduceat(exp_u, pt_starts_v)
            p = exp_u / np.repeat(sum_exp, pt_lens_v)

            yp = y_v * p
            S_g = np.add.reduceat(yp, pt_starts_v)
            loss = -np.mean(np.log(np.maximum(S_g, 1e-12))) + lmb * np.sum(beta ** 2)

            S_g_rep = np.repeat(S_g, pt_lens_v)
            w = p - (yp / np.maximum(S_g_rep, 1e-12))
            grad = (X_v.T @ w) / G + 2.0 * lmb * beta
            return loss, grad

        b0 = np.zeros(X.shape[1])
        res = minimize(
            loss_and_grad, b0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter}
        )
        self.beta = res.x
        return time.time() - t0

    def predict_probs(self, df_val: pd.DataFrame) -> Tuple[np.ndarray, float]:
        t0 = time.time()
        X_raw = df_val[self.feature_cols].values
        X = self.scaler.transform(X_raw)
        u_score = X @ self.beta

        _, pt_starts, pt_lens = get_point_group_info(df_val)
        probs = normalize_probabilities_per_point(u_score, pt_starts, pt_lens, is_log_scale=True)
        infer_time = time.time() - t0
        return probs, infer_time


# ==============================================================================
# Model Training & Cross-Validation Orchestrator
# ==============================================================================

def run_model_pipeline(model_name: str, df_train: pd.DataFrame, df_test: pd.DataFrame, repo_root: Path):
    """
    Executes 5-fold CV, refits on full training set, evaluates on student_test,
    and saves all standardized deliverables.
    """
    print(f"\n{'='*70}\nSTARTING MODEL: {model_name}\n{'='*70}")
    out_dir = repo_root / "outputs" / "models" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 5-Fold Cross-Validation
    fold_metrics_list = []
    oof_probs = np.zeros(len(df_train), dtype=float)

    # Hyperparameters log
    hyperparams = {"model_name": model_name, "random_state": RANDOM_STATE}
    if model_name == "nearest_arc":
        hyperparams["sigma_m"] = SIGMA_EMISSION
    elif model_name == "gaussian_nb":
        hyperparams["log1p_features"] = LOG1P_COLS
    elif model_name == "conditional_logit":
        # Pre-tuned lambda selection across training folds
        hyperparams["lambda"] = 1e-3
        hyperparams["max_iter"] = 100

    print("Running Trip-Grouped 5-Fold Cross-Validation...")
    for fold in range(5):
        val_mask = (df_train["fold"] == fold).values
        train_mask = ~val_mask

        df_tr = df_train[train_mask].reset_index(drop=True)
        df_val = df_train[val_mask].reset_index(drop=True)

        if model_name == "nearest_arc":
            model = NearestArcModel(sigma=SIGMA_EMISSION)
            train_time = 0.0
            probs_val, infer_time = model.predict_probs(df_val)

        elif model_name == "gaussian_nb":
            model = Log1pGaussianNBModel()
            train_time = model.fit(df_tr)
            probs_val, infer_time = model.predict_probs(df_val)

        elif model_name == "conditional_logit":
            model = ConditionalLogitModel(lmb=hyperparams["lambda"], max_iter=hyperparams["max_iter"])
            train_time = model.fit(df_tr)
            probs_val, infer_time = model.predict_probs(df_val)

        oof_probs[val_mask] = probs_val
        metrics, _, _ = evaluate_predictions(df_val, probs_val, infer_time, train_time)
        metrics["fold"] = fold
        fold_metrics_list.append(metrics)

        print(
            f"  Fold {fold}: Top-1 Acc={metrics['top1_acc']*100:.2f}% | "
            f"Macro F1={metrics['macro_f1']:.4f} | LogLoss={metrics['log_loss']:.4f} | "
            f"Train={metrics['train_time_s']:.2f}s | Infer={metrics['infer_ms_per_1k_rows']:.2f}ms/1k"
        )

    # Compile CV metrics dataframe
    cv_df = pd.DataFrame(fold_metrics_list)
    cols_order = [
        "fold", "top1_acc", "macro_f1", "precision_1", "recall_1",
        "log_loss", "train_time_s", "infer_ms_per_1k_rows", "n_rows"
    ]
    cv_df = cv_df[cols_order]
    cv_df.to_csv(out_dir / "cv_folds.csv", index=False)
    print(f"Saved CV folds to {out_dir / 'cv_folds.csv'}")

    # Calculate CV Mean & Std
    cv_mean = cv_df.mean().to_dict()
    cv_std = cv_df.std().to_dict()

    print(
        f"\nCV Mean (std): Top-1 Acc={cv_mean['top1_acc']*100:.2f}% (+/-{cv_std['top1_acc']*100:.2f}%) | "
        f"Macro F1={cv_mean['macro_f1']:.4f} (+/-{cv_std['macro_f1']:.4f}) | "
        f"LogLoss={cv_mean['log_loss']:.4f}"
    )

    # Save Out-Of-Fold Predictions
    oof_df = df_train[["trip_id", "point_idx", "arc_id", "cand_rank", "label", "fold"]].copy()
    oof_df["prob"] = oof_probs
    oof_df.to_parquet(out_dir / "oof_train.parquet", index=False)
    print(f"Saved OOF predictions to {out_dir / 'oof_train.parquet'}")

    # 2. Refit on Full Train Set & Evaluate on Student Test
    print("\nRefitting on ALL usable train rows and evaluating on student_test...")
    if model_name == "nearest_arc":
        final_model = NearestArcModel(sigma=SIGMA_EMISSION)
        final_train_time = 0.0
        test_probs, test_infer_time = final_model.predict_probs(df_test)

    elif model_name == "gaussian_nb":
        final_model = Log1pGaussianNBModel()
        final_train_time = final_model.fit(df_train)
        test_probs, test_infer_time = final_model.predict_probs(df_test)

    elif model_name == "conditional_logit":
        final_model = ConditionalLogitModel(lmb=hyperparams["lambda"], max_iter=hyperparams["max_iter"])
        final_train_time = final_model.fit(df_train)
        test_probs, test_infer_time = final_model.predict_probs(df_test)

        # Save interpretable model coefficients
        coef_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "coefficient": final_model.beta
        }).sort_values("coefficient", ascending=False)
        coef_df.to_csv(out_dir / "coefficients.csv", index=False)
        print("Top 5 Positive Features:", coef_df.head(5)[["feature", "coefficient"]].to_dict(orient="records"))
        print("Top 5 Negative Features:", coef_df.tail(5)[["feature", "coefficient"]].to_dict(orient="records"))

    # Test Metrics
    test_metrics, _, test_cm = evaluate_predictions(
        df_test, test_probs, test_infer_time, final_train_time
    )
    print(
        f"Student Test Results: Top-1 Acc={test_metrics['top1_acc']*100:.2f}% | "
        f"Macro F1={test_metrics['macro_f1']:.4f} | Precision={test_metrics['precision_1']:.4f} | "
        f"Recall={test_metrics['recall_1']:.4f} | LogLoss={test_metrics['log_loss']:.4f} | "
        f"Infer={test_metrics['infer_ms_per_1k_rows']:.2f}ms/1k"
    )

    # Save Student Test Predictions
    test_pred_df = df_test[["trip_id", "point_idx", "arc_id", "cand_rank", "label"]].copy()
    test_pred_df["prob"] = test_probs
    test_pred_df.to_parquet(out_dir / "pred_student_test.parquet", index=False)
    print(f"Saved test predictions to {out_dir / 'pred_student_test.parquet'}")

    # Save Confusion Matrix Plot
    plot_and_save_confusion_matrix(test_cm, out_dir / "confusion_matrix.png", model_name)
    print(f"Saved confusion matrix plot to {out_dir / 'confusion_matrix.png'}")

    # Save Final Model Artifact
    joblib.dump(final_model, out_dir / "model.joblib")
    print(f"Saved model artifact to {out_dir / 'model.joblib'}")

    # Save summary.json
    summary = {
        "model_name": model_name,
        "n_train_rows": len(df_train),
        "n_test_rows": len(df_test),
        "hyperparameters": hyperparams,
        "cv_summary": {
            "top1_acc_mean": round(cv_mean["top1_acc"], 6),
            "top1_acc_std": round(cv_std["top1_acc"], 6),
            "macro_f1_mean": round(cv_mean["macro_f1"], 6),
            "macro_f1_std": round(cv_std["macro_f1"], 6),
            "precision_1_mean": round(cv_mean["precision_1"], 6),
            "precision_1_std": round(cv_std["precision_1"], 6),
            "recall_1_mean": round(cv_mean["recall_1"], 6),
            "recall_1_std": round(cv_std["recall_1"], 6),
            "log_loss_mean": round(cv_mean["log_loss"], 6),
            "log_loss_std": round(cv_std["log_loss"], 6),
            "train_time_s_mean": round(cv_mean["train_time_s"], 4),
            "infer_ms_per_1k_rows_mean": round(cv_mean["infer_ms_per_1k_rows"], 4)
        },
        "student_test_metrics": test_metrics
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to {out_dir / 'summary.json'}")

    return summary


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="PathSnap Model Training & Evaluation (Baselines & Linear)")
    parser.add_argument(
        "--model",
        choices=["nearest_arc", "gaussian_nb", "conditional_logit", "all"],
        default="all",
        help="Which model to train and evaluate"
    )
    args = parser.parse_args()

    df_train, df_test = load_and_preprocess_data(REPO_ROOT)

    models_to_run = (
        ["nearest_arc", "gaussian_nb", "conditional_logit"]
        if args.model == "all"
        else [args.model]
    )

    summaries = {}
    for m in models_to_run:
        summaries[m] = run_model_pipeline(m, df_train, df_test, REPO_ROOT)

    print("\n" + "=" * 80)
    print("ALL REQUESTED MODELS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    for m, s in summaries.items():
        cv = s["cv_summary"]
        ts = s["student_test_metrics"]
        print(
            f"[{m.upper():<18}] "
            f"CV Top-1: {cv['top1_acc_mean']*100:.2f}% | "
            f"CV Macro F1: {cv['macro_f1_mean']:.4f} | "
            f"Test Top-1: {ts['top1_acc']*100:.2f}% | "
            f"Test Macro F1: {ts['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
