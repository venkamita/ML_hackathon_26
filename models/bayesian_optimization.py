"""
PathSnap: Bayesian Hyperparameter Optimisation (Section 4 Deliverables).
Implements Gaussian Process Bayesian Optimisation with Expected Improvement (EI)
on group-aware cross-validation on the 80 training trips.

Focuses on the chosen top-performing model: Conditional Logit (and optionally Gaussian NB & Nearest Arc).
Answers all 6 compulsory Section 4 questions:
  1. Hyperparameters included in search
  2. Search space ranges & distributions
  3. Objective function optimised (CV Macro F1)
  4. Number of optimisation trials
  5. Best hyperparameter values obtained
  6. Comprehensive Untuned vs. Tuned comparison table
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.build_candidates import FEATURE_COLS
from models.baselines_linear import (
    load_and_preprocess_data,
    get_point_group_info,
    normalize_probabilities_per_point,
    evaluate_predictions,
    ConditionalLogitModel,
    Log1pGaussianNBModel,
    NearestArcModel,
    RANDOM_STATE
)


class BayesianOptimizer1D:
    """
    Gaussian Process Bayesian Optimisation with Expected Improvement (EI) acquisition.
    Surrogate: GP with Matérn 5/2 kernel.
    """
    def __init__(self, bounds: Tuple[float, float], n_init: int = 4, xi: float = 0.01, random_state: int = RANDOM_STATE):
        self.bounds = bounds
        self.n_init = n_init
        self.xi = xi
        self.rng = np.random.RandomState(random_state)
        self.X_sample: List[float] = []
        self.y_sample: List[float] = []

    def get_next_candidate(self) -> float:
        # Initial exploration phase
        if len(self.X_sample) < self.n_init:
            val = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * (len(self.X_sample) / max(self.n_init - 1, 1))
            # Add small jitter to avoid duplicate exact points
            jitter = self.rng.uniform(-0.05, 0.05)
            return float(np.clip(val + jitter, self.bounds[0], self.bounds[1]))

        X = np.array(self.X_sample).reshape(-1, 1)
        y = np.array(self.y_sample)

        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-5,
            normalize_y=True,
            random_state=RANDOM_STATE
        )
        gp.fit(X, y)

        # Dense candidate grid
        grid = np.linspace(self.bounds[0], self.bounds[1], 200).reshape(-1, 1)
        mu, sigma = gp.predict(grid, return_std=True)

        best_y = np.max(y)
        improvement = mu - best_y - self.xi
        Z = np.where(sigma > 1e-9, improvement / sigma, 0.0)
        ei = np.where(sigma > 1e-9, improvement * norm.cdf(Z) + sigma * norm.pdf(Z), 0.0)

        # Pick candidate maximizing Expected Improvement
        best_idx = int(np.argmax(ei))
        return float(grid[best_idx, 0])

    def register(self, x: float, y: float):
        self.X_sample.append(x)
        self.y_sample.append(y)


def evaluate_conditional_logit_cv(df_train: pd.DataFrame, lmb: float) -> Tuple[float, float, float]:
    """Evaluates 5-fold CV for a given lambda. Returns (mean_macro_f1, mean_top1_acc, mean_log_loss)."""
    f1_list, top1_list, loss_list = [], [], []

    for fold in range(5):
        val_mask = (df_train["fold"] == fold).values
        train_mask = ~val_mask

        df_tr = df_train[train_mask].reset_index(drop=True)
        df_val = df_train[val_mask].reset_index(drop=True)

        model = ConditionalLogitModel(lmb=lmb, max_iter=80)
        model.fit(df_tr)
        probs, infer_time = model.predict_probs(df_val)

        metrics, _, _ = evaluate_predictions(df_val, probs, infer_time, 0.0)
        f1_list.append(metrics["macro_f1"])
        top1_list.append(metrics["top1_acc"])
        loss_list.append(metrics["log_loss"])

    return float(np.mean(f1_list)), float(np.mean(top1_list)), float(np.mean(loss_list))


def run_bayesian_optimization_conditional_logit(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_trials: int = 12
) -> Dict[str, Any]:
    """
    Executes full Bayesian Optimisation for the chosen Conditional Logit model.
    Search Space: log10(lambda) in [-6.0, 0.0] (lambda in [1e-6, 1.0]).
    Objective: 5-Fold Grouped CV Macro F1-score.
    """
    out_dir = REPO_ROOT / "outputs" / "models" / "conditional_logit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("BAYESIAN HYPERPARAMETER OPTIMISATION: CONDITIONAL LOGIT")
    print("=" * 75)
    print(f"Objective Function: 5-Fold Group-Aware CV Macro F1-score (80 training trips)")
    print(f"Hyperparameter: L2 regularization strength lambda")
    print(f"Search Distribution: Log-uniform, log10(lambda) in [-6.0, 0.0] (1e-6 to 1.0)")
    print(f"Total Trials: {n_trials}\n")

    optimizer = BayesianOptimizer1D(bounds=(-6.0, 0.0), n_init=4)
    trials_records = []

    for trial_idx in range(1, n_trials + 1):
        t0 = time.time()
        log_lmb = optimizer.get_next_candidate()
        lmb = float(10.0 ** log_lmb)

        cv_f1, cv_top1, cv_loss = evaluate_conditional_logit_cv(df_train, lmb)
        trial_time = time.time() - t0

        optimizer.register(log_lmb, cv_f1)
        trials_records.append({
            "trial": trial_idx,
            "log10_lambda": round(log_lmb, 4),
            "lambda": f"{lmb:.2e}",
            "cv_macro_f1": round(cv_f1, 6),
            "cv_top1_acc": round(cv_top1, 6),
            "cv_log_loss": round(cv_loss, 6),
            "duration_s": round(trial_time, 2)
        })

        print(
            f"Trial {trial_idx:02d}/{n_trials:02d} | lambda={lmb:.2e} (log10={log_lmb:.2f}) | "
            f"CV Macro F1 = {cv_f1:.4f} | CV Top-1 = {cv_top1*100:.2f}% | "
            f"CV LogLoss = {cv_loss:.4f} ({trial_time:.1f}s)"
        )

    df_trials = pd.DataFrame(trials_records)
    trials_csv = out_dir / "bayes_opt_trials.csv"
    df_trials.to_csv(trials_csv, index=False)
    print(f"\nSaved all {n_trials} Bayesian optimization trials to {trials_csv}")

    # Best Hyperparameter
    best_trial = df_trials.loc[df_trials["cv_macro_f1"].idxmax()]
    best_log_lmb = best_trial["log10_lambda"]
    best_lmb = float(10.0 ** best_log_lmb)
    print(f"\n>>> Best Hyperparameter: lambda = {best_lmb:.2e} (CV Macro F1 = {best_trial['cv_macro_f1']:.4f})")

    # Plot Convergence Curve
    plt.figure(figsize=(8, 4.5))
    trials_arr = df_trials["trial"].values
    f1_arr = df_trials["cv_macro_f1"].values
    best_so_far = np.maximum.accumulate(f1_arr)

    plt.plot(trials_arr, f1_arr, "o--", color="#1f77b4", label="Trial Score (CV Macro F1)")
    plt.plot(trials_arr, best_so_far, "s-", color="#2ca02c", linewidth=2, label="Best Score So Far")
    plt.axvline(best_trial["trial"], color="red", linestyle=":", label=f"Best Trial ({best_trial['trial']})")
    plt.xlabel("Bayesian Optimisation Trial Number", fontsize=11)
    plt.ylabel("Validation Macro F1-score", fontsize=11)
    plt.title("Bayesian Hyperparameter Optimisation: Convergence History (Conditional Logit)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    conv_plot = out_dir / "bayes_opt_convergence.png"
    plt.savefig(conv_plot, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {conv_plot}")

    # Refit Tuned Model on ALL 80 Training Trips
    print("\nRefitting Tuned Model (best lambda) on full training set...")
    tuned_model = ConditionalLogitModel(lmb=best_lmb, max_iter=100)
    tuned_train_time = tuned_model.fit(df_train)
    tuned_test_probs, tuned_test_infer = tuned_model.predict_probs(df_test)

    tuned_test_metrics, _, _ = evaluate_predictions(
        df_test, tuned_test_probs, tuned_test_infer, tuned_train_time
    )

    # Load Untuned Baseline Metrics from summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        untuned_summary = json.load(f)

    untuned_cv = untuned_summary["cv_summary"]
    untuned_test = untuned_summary["student_test_metrics"]

    # Build Section 4 Compulsory Comparison Table
    comparison_rows = [
        {
            "Metric": "Cross-validation Macro F1-score",
            "Untuned (lambda=1e-3)": round(untuned_cv["macro_f1_mean"], 4),
            "Tuned (lambda=" + f"{best_lmb:.2e}" + ")": round(best_trial["cv_macro_f1"], 4),
            "Absolute Delta": round(best_trial["cv_macro_f1"] - untuned_cv["macro_f1_mean"], 4)
        },
        {
            "Metric": "Student-testing Top-1 Accuracy",
            "Untuned (lambda=1e-3)": f"{untuned_test['top1_acc']*100:.2f}%",
            "Tuned (lambda=" + f"{best_lmb:.2e}" + ")": f"{tuned_test_metrics['top1_acc']*100:.2f}%",
            "Absolute Delta": f"{(tuned_test_metrics['top1_acc'] - untuned_test['top1_acc'])*100:+.2f}%"
        },
        {
            "Metric": "Student-testing Macro F1-score",
            "Untuned (lambda=1e-3)": round(untuned_test["macro_f1"], 4),
            "Tuned (lambda=" + f"{best_lmb:.2e}" + ")": round(tuned_test_metrics["macro_f1"], 4),
            "Absolute Delta": round(tuned_test_metrics["macro_f1"] - untuned_test["macro_f1"], 4)
        },
        {
            "Metric": "Log Loss (Student-testing)",
            "Untuned (lambda=1e-3)": round(untuned_test["log_loss"], 4),
            "Tuned (lambda=" + f"{best_lmb:.2e}" + ")": round(tuned_test_metrics["log_loss"], 4),
            "Absolute Delta": round(tuned_test_metrics["log_loss"] - untuned_test["log_loss"], 4)
        },
        {
            "Metric": "Inference Time (ms / 1k rows)",
            "Untuned (lambda=1e-3)": round(untuned_test["infer_ms_per_1k_rows"], 2),
            "Tuned (lambda=" + f"{best_lmb:.2e}" + ")": round(tuned_test_metrics["infer_ms_per_1k_rows"], 2),
            "Absolute Delta": round(tuned_test_metrics["infer_ms_per_1k_rows"] - untuned_test["infer_ms_per_1k_rows"], 2)
        }
    ]

    df_comp = pd.DataFrame(comparison_rows)
    comp_csv = out_dir / "untuned_vs_tuned_comparison.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"Saved comparison table to {comp_csv}")

    print("\n" + "=" * 75)
    print("SECTION 4: UNTUNED VS TUNED MODEL COMPARISON TABLE")
    print("=" * 75)
    print(df_comp.to_string(index=False))

    return {
        "best_lambda": best_lmb,
        "best_cv_macro_f1": best_trial["cv_macro_f1"],
        "comparison_table": comparison_rows
    }


def main():
    parser = argparse.ArgumentParser(description="PathSnap Bayesian Hyperparameter Optimisation")
    parser.add_argument("--trials", type=int, default=10, help="Number of Bayesian optimization trials")
    args = parser.parse_args()

    df_train, df_test = load_and_preprocess_data(REPO_ROOT)
    run_bayesian_optimization_conditional_logit(df_train, df_test, n_trials=args.trials)


if __name__ == "__main__":
    main()
