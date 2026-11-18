from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models.eval_utils import (
    RANDOM_STATE,
    calculate_metrics,
    inner_validation_trips,
    load_usable,
    make_oof,
    mean_std_metrics,
    per_point_probabilities,
    save_confusion_matrix,
    save_learning_curves,
    trip_folds,
)


DEFAULT_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


def point_groups(df: pd.DataFrame) -> np.ndarray:
    return df.groupby(["trip_id", "point_idx"], sort=False).size().to_numpy()


def fit_temperature(raw_scores: np.ndarray, df: pd.DataFrame) -> float:
    y = df["label"].to_numpy(dtype=float)

    def objective(temperature: float) -> float:
        p = per_point_probabilities(df, raw_scores, temperature=temperature, real_valued=True)
        from sklearn.metrics import log_loss
        return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))

    result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def binary_params(extra: dict | None = None) -> dict:
    params = dict(DEFAULT_PARAMS)
    params.update({"objective": "binary", "metric": "binary_logloss"})
    if extra:
        params.update(extra)
    return params


def rank_params(extra: dict | None = None) -> dict:
    params = dict(DEFAULT_PARAMS)
    params.update({"objective": "lambdarank", "metric": "ndcg", "eval_at": [1]})
    if extra:
        params.update(extra)
    return params


def make_ranker(params: dict) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(**params)


def make_classifier(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**params)


def train_binary(train: pd.DataFrame, test: pd.DataFrame, extra_params: dict | None = None,
                 output_name: str = "lgbm_binary"):
    out = Path("outputs/models") / output_name
    out.mkdir(parents=True, exist_ok=True)
    folds = trip_folds(train["trip_id"])
    train = train.copy()
    train["fold"] = train["trip_id"].map(folds)
    oof_prob = np.empty(len(train), dtype=float)
    fold_rows = []
    learning_curves = {}

    for fold in range(5):
        fit_trips = sorted(train.loc[train["fold"] != fold, "trip_id"].unique())
        inner_fit, inner_val = inner_validation_trips(fit_trips)
        fit_mask = train["trip_id"].isin(inner_fit)
        val_mask = train["trip_id"].isin(inner_val)
        val_outer_mask = train["fold"] == fold
        params = binary_params(extra_params)
        model = make_classifier(params)
        started = time.perf_counter()
        model.fit(
            train.loc[fit_mask, FEATURE_COLS], train.loc[fit_mask, "label"],
            eval_set=[(train.loc[val_mask, FEATURE_COLS], train.loc[val_mask, "label"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        train_time = time.perf_counter() - started
        raw = model.predict_proba(train.loc[val_outer_mask, FEATURE_COLS])[:, 1]
        oof_prob[val_outer_mask.to_numpy()] = per_point_probabilities(
            train.loc[val_outer_mask], raw, real_valued=False
        )
        metrics = calculate_metrics(
            train.loc[val_outer_mask].reset_index(drop=True),
            oof_prob[val_outer_mask.to_numpy()],
        )
        started = time.perf_counter()
        infer_scores = model.predict_proba(train.loc[val_outer_mask, FEATURE_COLS])[:, 1]
        infer_time = (time.perf_counter() - started) * 1000 / max(len(infer_scores), 1) * 1000
        metrics.update({
            "fold": fold,
            "train_time_s": train_time,
            "infer_ms_per_1k_rows": infer_time,
            "n_train_rows": int(fit_mask.sum()),
            "n_inner_val_trips": len(inner_val),
        })
        fold_rows.append(metrics)
        learning_curves[str(fold)] = model.evals_result_

    pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
    oof = make_oof(train.drop(columns=["fold"]), oof_prob, train["fold"].to_numpy())
    oof.to_parquet(out / "oof_train.parquet", index=False)

    for fold, result in learning_curves.items():
        save_learning_curves(result, out / f"learning_curve_fold_{fold}.png",
                             f"LightGBM binary learning curve - fold {fold}")

    _, test_usable = load_usable("outputs/features/features_student_test.parquet")
    fit_trips = sorted(train["trip_id"].unique())
    inner_fit, inner_val = inner_validation_trips(fit_trips)
    fit_mask = train["trip_id"].isin(inner_fit)
    val_mask = train["trip_id"].isin(inner_val)
    final_params = binary_params(extra_params)
    final_model = make_classifier(final_params)
    final_started = time.perf_counter()
    final_model.fit(
        train.loc[fit_mask, FEATURE_COLS], train.loc[fit_mask, "label"],
        eval_set=[(train.loc[val_mask, FEATURE_COLS], train.loc[val_mask, "label"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    final_fit_time = time.perf_counter() - final_started
    test_raw = final_model.predict_proba(test_usable[FEATURE_COLS])[:, 1]
    test_prob = per_point_probabilities(test_usable, test_raw, real_valued=False)
    test_metrics = calculate_metrics(test_usable.reset_index(drop=True), test_prob)
    infer_started = time.perf_counter()
    final_model.predict(test_usable[FEATURE_COLS])
    test_infer_ms = (time.perf_counter() - infer_started) * 1000 / len(test_usable) * 1000

    pred = pd.DataFrame({
        "trip_id": test_usable["trip_id"].to_numpy(),
        "point_idx": test_usable["point_idx"].to_numpy(),
        "arc_id": test_usable["arc_id"].to_numpy(),
        "cand_rank": test_usable["cand_rank"].to_numpy(),
        "label": test_usable["label"].to_numpy(),
        "prob": test_prob,
    })
    pred.to_parquet(out / "pred_student_test.parquet", index=False)

    cm = [[test_metrics["cm_tn"], test_metrics["cm_fp"]],
          [test_metrics["cm_fn"], test_metrics["cm_tp"]]]
    save_confusion_matrix(cm, out / "confusion_matrix.png", "LightGBM binary - student test")

    final_params_all = binary_params(extra_params)
    final_all = make_classifier(final_params_all)
    final_all.fit(train[FEATURE_COLS], train["label"])
    joblib.dump({
        "model": final_all,
        "model_type": "lgbm_binary",
        "feature_cols": FEATURE_COLS,
        "probability_mode": "point_normalised_binary",
    }, out / "model.joblib")

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": final_all.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)

    metric_names = ["top1_acc", "macro_f1", "precision_1", "recall_1", "log_loss",
                    "train_time_s", "infer_ms_per_1k_rows"]
    summary = mean_std_metrics(fold_rows, metric_names)
    summary.update({
        "model_name": output_name,
        "student_test_metrics": test_metrics,
        "student_test_infer_ms_per_1k_rows": test_infer_ms,
        "hyperparameters": final_params_all,
        "n_train_rows": int(len(train)),
        "n_train_points": int(train.groupby(["trip_id", "point_idx"]).ngroups),
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "learning_curves.json").write_text(
        json.dumps({str(k): v for k, v in learning_curves.items()}, indent=2), encoding="utf-8"
    )
    return summary


def train_rank(train: pd.DataFrame, test: pd.DataFrame, extra_params: dict | None = None,
               output_name: str = "lgbm_rank"):
    out = Path("outputs/models") / output_name
    out.mkdir(parents=True, exist_ok=True)
    folds = trip_folds(train["trip_id"])
    train = train.copy()
    train["fold"] = train["trip_id"].map(folds)
    oof_raw = np.empty(len(train), dtype=float)
    fold_rows = []
    learning_curves = {}

    for fold in range(5):
        outer_fit_trips = sorted(train.loc[train["fold"] != fold, "trip_id"].unique())
        inner_fit, inner_val = inner_validation_trips(outer_fit_trips)
        fit_mask = train["trip_id"].isin(inner_fit)
        val_mask = train["trip_id"].isin(inner_val)
        val_outer_mask = train["fold"] == fold
        params = rank_params(extra_params)
        model = make_ranker(params)
        fit_df = train.loc[fit_mask]
        val_df = train.loc[val_mask]
        started = time.perf_counter()
        model.fit(
            fit_df[FEATURE_COLS], fit_df["label"],
            group=point_groups(fit_df),
            eval_set=[(val_df[FEATURE_COLS], val_df["label"])],
            eval_group=[point_groups(val_df)],
            eval_metric="ndcg",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        train_time = time.perf_counter() - started
        raw = model.predict(train.loc[val_outer_mask, FEATURE_COLS])
        oof_raw[val_outer_mask.to_numpy()] = raw

        started = time.perf_counter()
        infer_raw = model.predict(train.loc[val_outer_mask, FEATURE_COLS])
        infer_time = (time.perf_counter() - started) * 1000 / max(len(infer_raw), 1) * 1000
        fold_rows.append({
            "fold": fold,
            "train_time_s": train_time,
            "infer_ms_per_1k_rows": infer_time,
            "n_train_rows": int(fit_mask.sum()),
            "n_inner_val_trips": len(inner_val),
        })
        learning_curves[str(fold)] = model.evals_result_

    temperature = fit_temperature(oof_raw, train)
    oof_prob = per_point_probabilities(train, oof_raw, temperature=temperature, real_valued=True)
    for row in fold_rows:
        fold_mask = train["fold"].to_numpy() == row["fold"]
        fold_df = train.loc[fold_mask].reset_index(drop=True)
        fold_prob = oof_prob[fold_mask]
        metrics = calculate_metrics(fold_df, fold_prob)
        row.update(metrics)

    pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
    oof = make_oof(train.drop(columns=["fold"]), oof_prob, train["fold"].to_numpy())
    oof.to_parquet(out / "oof_train.parquet", index=False)
    for fold, result in learning_curves.items():
        save_learning_curves(result, out / f"learning_curve_fold_{fold}.png",
                             f"LightGBM LambdaRank learning curve - fold {fold}")

    _, test_usable = load_usable("outputs/features/features_student_test.parquet")
    fit_trips = sorted(train["trip_id"].unique())
    inner_fit, inner_val = inner_validation_trips(fit_trips)
    fit_mask = train["trip_id"].isin(inner_fit)
    val_mask = train["trip_id"].isin(inner_val)
    fit_df = train.loc[fit_mask]
    val_df = train.loc[val_mask]
    final_model = make_ranker(rank_params(extra_params))
    final_model.fit(
        fit_df[FEATURE_COLS], fit_df["label"], group=point_groups(fit_df),
        eval_set=[(val_df[FEATURE_COLS], val_df["label"])],
        eval_group=[point_groups(val_df)],
        eval_metric="ndcg",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    test_raw = final_model.predict(test_usable[FEATURE_COLS])
    test_prob = per_point_probabilities(test_usable, test_raw, temperature=temperature,
                                        real_valued=True)
    test_metrics = calculate_metrics(test_usable.reset_index(drop=True), test_prob)
    infer_started = time.perf_counter()
    final_model.predict(test_usable[FEATURE_COLS])
    test_infer_ms = (time.perf_counter() - infer_started) * 1000 / len(test_usable) * 1000

    pred = pd.DataFrame({
        "trip_id": test_usable["trip_id"].to_numpy(),
        "point_idx": test_usable["point_idx"].to_numpy(),
        "arc_id": test_usable["arc_id"].to_numpy(),
        "cand_rank": test_usable["cand_rank"].to_numpy(),
        "label": test_usable["label"].to_numpy(),
        "prob": test_prob,
    })
    pred.to_parquet(out / "pred_student_test.parquet", index=False)
    cm = [[test_metrics["cm_tn"], test_metrics["cm_fp"]],
          [test_metrics["cm_fn"], test_metrics["cm_tp"]]]
    save_confusion_matrix(cm, out / "confusion_matrix.png", "LightGBM LambdaRank - student test")

    final_all = make_ranker(rank_params(extra_params))
    final_all.fit(train[FEATURE_COLS], train["label"], group=point_groups(train))
    joblib.dump({
        "model": final_all,
        "model_type": "lgbm_rank",
        "feature_cols": FEATURE_COLS,
        "probability_mode": "softmax",
        "temperature": temperature,
    }, out / "model.joblib")
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": final_all.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)

    metric_names = ["top1_acc", "macro_f1", "precision_1", "recall_1", "log_loss",
                    "train_time_s", "infer_ms_per_1k_rows"]
    summary = mean_std_metrics(fold_rows, metric_names)
    summary.update({
        "model_name": output_name,
        "temperature": temperature,
        "student_test_metrics": test_metrics,
        "student_test_infer_ms_per_1k_rows": test_infer_ms,
        "hyperparameters": rank_params(extra_params),
        "n_train_rows": int(len(train)),
        "n_train_points": int(train.groupby(["trip_id", "point_idx"]).ngroups),
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "learning_curves.json").write_text(
        json.dumps({str(k): v for k, v in learning_curves.items()}, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["binary", "rank", "both"], default="binary")
    args = parser.parse_args()
    _, train = load_usable("outputs/features/features_train.parquet")
    _, test = load_usable("outputs/features/features_student_test.parquet")
    if args.variant in ("binary", "both"):
        print(json.dumps(train_binary_summary(train, test), indent=2))
    if args.variant in ("rank", "both"):
        print(json.dumps(train_rank_summary(train, test), indent=2))


def train_binary_summary(train, test):
    return train_binary(train, test)


def train_rank_summary(train, test):
    return train_rank(train, test)


if __name__ == "__main__":
    main()
