"""
Evaluate all three models on the newly provided evaluator_test dataset:
  (0) nearest_arc
  (1) gaussian_nb
  (2) conditional_logit (Tuned: lambda = 1.05e-02, and Untuned: lambda = 1e-3)
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.build_candidates import FEATURE_COLS
from models.baselines_linear import (
    evaluate_predictions,
    plot_and_save_confusion_matrix,
    NearestArcModel,
    Log1pGaussianNBModel,
    ConditionalLogitModel,
    SIGMA_EMISSION
)


def run_evaluator_test():
    test_path = REPO_ROOT / "outputs" / "features" / "features_evaluator_test.parquet"
    train_path = REPO_ROOT / "outputs" / "features" / "features_train.parquet"

    print(f"Loading evaluator test data from {test_path}...")
    df_eval_raw = pd.read_parquet(test_path)
    df_eval = (
        df_eval_raw[df_eval_raw["label_usable"] == 1]
        .sort_values(["trip_id", "point_idx", "cand_rank"])
        .reset_index(drop=True)
    )

    print(f"Loading training data from {train_path} for final model refits...")
    df_train_raw = pd.read_parquet(train_path)
    df_train = (
        df_train_raw[df_train_raw["label_usable"] == 1]
        .sort_values(["trip_id", "point_idx", "cand_rank"])
        .reset_index(drop=True)
    )

    print(f"\nEvaluator Test Set:")
    print(f"  Total Usable Rows: {len(df_eval):,}")
    print(f"  Unique Trips: {df_eval['trip_id'].nunique()} ({sorted(df_eval['trip_id'].unique())})")
    print(f"  Unique GPS Points: {df_eval.groupby(['trip_id', 'point_idx']).ngroups:,}")
    print(f"  Positive Rate: {df_eval['label'].mean()*100:.2f}% (sum={df_eval['label'].sum():,})\n")

    results_table = []

    # ---------------------------------------------------------
    # 0. Nearest Arc Baseline
    # ---------------------------------------------------------
    print("=" * 60)
    print("Evaluating Model 0: nearest_arc ...")
    model_0 = NearestArcModel(sigma=SIGMA_EMISSION)
    probs_0, infer_time_0 = model_0.predict_probs(df_eval)
    metrics_0, _, cm_0 = evaluate_predictions(df_eval, probs_0, infer_time_0, 0.0)

    out_dir_0 = REPO_ROOT / "outputs" / "models" / "nearest_arc"
    out_dir_0.mkdir(parents=True, exist_ok=True)
    pred_0 = df_eval[["trip_id", "point_idx", "arc_id", "cand_rank", "label"]].copy()
    pred_0["prob"] = probs_0
    pred_0.to_parquet(out_dir_0 / "pred_evaluator_test.parquet", index=False)
    plot_and_save_confusion_matrix(cm_0, out_dir_0 / "evaluator_test_confusion_matrix.png", "nearest_arc (Evaluator Test)")

    with open(out_dir_0 / "evaluator_test_metrics.json", "w") as f:
        json.dump(metrics_0, f, indent=2)

    results_table.append({
        "Model": "nearest_arc (Baseline)",
        "Top-1 Accuracy": f"{metrics_0['top1_acc']*100:.2f}%",
        "Macro F1": round(metrics_0["macro_f1"], 4),
        "Precision (1)": round(metrics_0["precision_1"], 4),
        "Recall (1)": round(metrics_0["recall_1"], 4),
        "Log Loss": round(metrics_0["log_loss"], 4),
        "Latency (ms/1k)": round(metrics_0["infer_ms_per_1k_rows"], 2)
    })
    print(f"nearest_arc -> Top-1: {metrics_0['top1_acc']*100:.2f}% | Macro F1: {metrics_0['macro_f1']:.4f} | LogLoss: {metrics_0['log_loss']:.4f}")

    # ---------------------------------------------------------
    # 2. Gaussian Naive Bayes
    # ---------------------------------------------------------
    print("=" * 60)
    print("Evaluating Model 2: gaussian_nb ...")
    model_2 = Log1pGaussianNBModel()
    train_time_2 = model_2.fit(df_train)
    probs_2, infer_time_2 = model_2.predict_probs(df_eval)
    metrics_2, _, cm_2 = evaluate_predictions(df_eval, probs_2, infer_time_2, train_time_2)

    out_dir_2 = REPO_ROOT / "outputs" / "models" / "gaussian_nb"
    out_dir_2.mkdir(parents=True, exist_ok=True)
    pred_2 = df_eval[["trip_id", "point_idx", "arc_id", "cand_rank", "label"]].copy()
    pred_2["prob"] = probs_2
    pred_2.to_parquet(out_dir_2 / "pred_evaluator_test.parquet", index=False)
    plot_and_save_confusion_matrix(cm_2, out_dir_2 / "evaluator_test_confusion_matrix.png", "gaussian_nb (Evaluator Test)")

    with open(out_dir_2 / "evaluator_test_metrics.json", "w") as f:
        json.dump(metrics_2, f, indent=2)

    results_table.append({
        "Model": "gaussian_nb",
        "Top-1 Accuracy": f"{metrics_2['top1_acc']*100:.2f}%",
        "Macro F1": round(metrics_2["macro_f1"], 4),
        "Precision (1)": round(metrics_2["precision_1"], 4),
        "Recall (1)": round(metrics_2["recall_1"], 4),
        "Log Loss": round(metrics_2["log_loss"], 4),
        "Latency (ms/1k)": round(metrics_2["infer_ms_per_1k_rows"], 2)
    })
    print(f"gaussian_nb -> Top-1: {metrics_2['top1_acc']*100:.2f}% | Macro F1: {metrics_2['macro_f1']:.4f} | LogLoss: {metrics_2['log_loss']:.4f}")

    # ---------------------------------------------------------
    # 1. Conditional Logit (Tuned: lambda = 1.05e-2)
    # ---------------------------------------------------------
    print("=" * 60)
    print("Evaluating Model 1: conditional_logit (Tuned: lambda = 1.05e-2) ...")
    model_1 = ConditionalLogitModel(lmb=1.05e-2, max_iter=100)
    train_time_1 = model_1.fit(df_train)
    probs_1, infer_time_1 = model_1.predict_probs(df_eval)
    metrics_1, _, cm_1 = evaluate_predictions(df_eval, probs_1, infer_time_1, train_time_1)

    out_dir_1 = REPO_ROOT / "outputs" / "models" / "conditional_logit"
    out_dir_1.mkdir(parents=True, exist_ok=True)
    pred_1 = df_eval[["trip_id", "point_idx", "arc_id", "cand_rank", "label"]].copy()
    pred_1["prob"] = probs_1
    pred_1.to_parquet(out_dir_1 / "pred_evaluator_test.parquet", index=False)
    plot_and_save_confusion_matrix(cm_1, out_dir_1 / "evaluator_test_confusion_matrix.png", "conditional_logit (Evaluator Test)")

    with open(out_dir_1 / "evaluator_test_metrics.json", "w") as f:
        json.dump(metrics_1, f, indent=2)

    results_table.append({
        "Model": "conditional_logit (Tuned)",
        "Top-1 Accuracy": f"{metrics_1['top1_acc']*100:.2f}%",
        "Macro F1": round(metrics_1["macro_f1"], 4),
        "Precision (1)": round(metrics_1["precision_1"], 4),
        "Recall (1)": round(metrics_1["recall_1"], 4),
        "Log Loss": round(metrics_1["log_loss"], 4),
        "Latency (ms/1k)": round(metrics_1["infer_ms_per_1k_rows"], 2)
    })
    print(f"conditional_logit -> Top-1: {metrics_1['top1_acc']*100:.2f}% | Macro F1: {metrics_1['macro_f1']:.4f} | LogLoss: {metrics_1['log_loss']:.4f}")

    # Compile Final Summary Table
    df_res = pd.DataFrame(results_table)
    summary_csv = REPO_ROOT / "outputs" / "models" / "evaluator_test_overall_comparison.csv"
    df_res.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("FINAL EVALUATOR TEST RESULTS (ALL 3 MODELS):")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print(f"\nSaved summary comparison to {summary_csv}")


if __name__ == "__main__":
    run_evaluator_test()
