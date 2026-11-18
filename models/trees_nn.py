"""Train and evaluate the PathSnap Extra Trees and MLP baselines.

Run from the repository root with scikit-learn and joblib installed:
    python -m models.trees_nn --model extra_trees
    python -m models.trees_nn --model mlp

The Extra Trees command regenerates outputs/models/extra_trees/model.joblib.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss
from sklearn.metrics import precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from features.build_candidates import FEATURE_COLS

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SEED = 42
N_FOLDS = 5
MLP_TRIPS = 25
MLP_MAX_EPOCHS = 40
MLP_PATIENCE = 5
TREE_FOLD_BUDGET_S = 300
POINT_KEYS = ["trip_id", "point_idx"]
PREDICTION_COLUMNS = ["trip_id", "point_idx", "arc_id", "cand_rank", "label"]
FEATURE_DIR = Path("outputs/features")
OUTPUT_DIR = Path("outputs/models")


class CandidateBatch:
    """A sorted set of rows containing whole candidate groups."""

    def __init__(self, rows, x):
        self.rows = rows
        self.x = x
        self.y = rows["label"].to_numpy(dtype=np.int8)
        trip, point = rows["trip_id"].to_numpy(), rows["point_idx"].to_numpy()
        change = (trip[1:] != trip[:-1]) | (point[1:] != point[:-1])
        self.starts = np.r_[0, np.flatnonzero(change) + 1]
        self.counts = np.diff(np.r_[self.starts, len(rows)])
        self.row_indices = np.arange(len(rows))

    def take(self, indices):
        return CandidateBatch(self.rows.iloc[indices], self.x[indices])

    def normalize(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != self.y.shape or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Expected one finite, nonnegative score per candidate.")
        totals = np.add.reduceat(scores, self.starts)
        # An all-zero point has no relative evidence, so give its candidates equal mass.
        uniform = np.repeat(1.0 / self.counts, self.counts)
        return np.divide(
            scores, np.repeat(np.where(totals > 0, totals, 1), self.counts),
            out=uniform, where=np.repeat(totals > 0, self.counts),
        )

    def evaluate(self, probs):
        max_per_point = np.maximum.reduceat(probs, self.starts)
        is_max = probs == np.repeat(max_per_point, self.counts)
        # The first row at each maximum has the lowest cand_rank.
        winners = np.minimum.reduceat(
            np.where(is_max, self.row_indices, len(probs)), self.starts,
        )
        pred = np.zeros(len(probs), dtype=np.int8)
        pred[winners] = 1
        tn, fp, fn, tp = confusion_matrix(self.y, pred, labels=[0, 1]).ravel()
        return {
            "top1_acc": float(self.y[winners].mean()),
            "macro_f1": float(f1_score(self.y, pred, average="macro")),
            "precision_1": float(precision_score(self.y, pred, zero_division=0)),
            "recall_1": float(recall_score(self.y, pred, zero_division=0)),
            "log_loss": self.log_loss(probs),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    def log_loss(self, probs):
        return float(log_loss(self.y, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1]))


def load_split(split):
    path = FEATURE_DIR / f"features_{split}.parquet"
    cols = list(dict.fromkeys([*PREDICTION_COLUMNS, "label_usable", *FEATURE_COLS]))
    frame = pd.read_parquet(path, columns=cols)
    n_cols = pq.ParquetFile(path).metadata.num_columns
    usable = frame.loc[frame["label_usable"].eq(1)].sort_values(
        [*POINT_KEYS, "cand_rank"], kind="stable", ignore_index=True,
    )
    if usable.empty or not usable["label"].isin([0, 1]).all():
        raise ValueError(f"{split} has no usable rows or has invalid usable labels.")
    x = usable[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise ValueError(f"{split} contains a nonfinite model feature.")
    counts = usable.groupby(POINT_KEYS, sort=False)["label"].sum()
    print(
        f"{split}: shape=({len(frame)}, {n_cols}), all_trips={frame.trip_id.nunique()}, "
        f"usable_rows={len(usable)}, usable_trips={usable.trip_id.nunique()}, "
        f"positive_rate_all={frame.label.mean():.6%}, positive_rate_usable={usable.label.mean():.6%}, "
        f"points={len(counts)}, no_positive={counts.eq(0).sum()}, "
        f"multiple_positive={counts.gt(1).sum()}", flush=True,
    )
    return CandidateBatch(usable[PREDICTION_COLUMNS], x)


def fold_ids(train):
    trips = sorted(train.rows["trip_id"].unique())
    mapping = {trip: i % N_FOLDS for i, trip in enumerate(trips)}
    folds = train.rows["trip_id"].map(mapping).to_numpy(dtype=np.int8)
    for fold in range(N_FOLDS):
        match = folds == fold
        val_trips = train.rows.loc[match, "trip_id"].nunique()
        print(
            f"fold={fold}: train_trips={len(trips) - val_trips}, val_trips={val_trips}, "
            f"val_rows={match.sum()}", flush=True,
        )
    return folds


def subsample_trips(batch):
    trips = sorted(batch.rows["trip_id"].unique())
    chosen = sorted(np.random.default_rng(SEED).choice(trips, size=MLP_TRIPS, replace=False))
    indices = np.flatnonzero(batch.rows["trip_id"].isin(chosen).to_numpy())
    return batch.take(indices), chosen


def predict(model, batch):
    start = perf_counter()
    probs = batch.normalize(model.predict_proba(batch.x)[:, 1])
    ms_per_1k = (perf_counter() - start) * 1_000_000 / len(batch.y)
    return probs, ms_per_1k


def train_mlp(train, validation=None, refit_epochs=None, hyperparameters=None):
    defaults = {
        "hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 1e-4,
        "batch_size": 1024, "learning_rate_init": 1e-3, "random_state": SEED,
    }
    model = MLPClassifier(**(defaults | (hyperparameters or {})))
    scaler = StandardScaler()
    start = perf_counter()
    x_train = scaler.fit_transform(train.x)
    fit_time = perf_counter() - start
    x_val = scaler.transform(validation.x) if validation is not None else None
    losses, best_loss, best_model, best_epoch = [], np.inf, None, 0
    epochs = MLP_MAX_EPOCHS if refit_epochs is None else refit_epochs

    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.partial_fit(x_train, train.y, classes=[0, 1])
        fit_time += perf_counter() - start
        if validation is None:
            continue

        train_probs = train.normalize(model.predict_proba(x_train)[:, 1])
        val_probs = validation.normalize(model.predict_proba(x_val)[:, 1])
        train_loss, val_loss = train.log_loss(train_probs), validation.log_loss(val_probs)
        losses.append({"epoch": epoch, "train_log_loss": train_loss, "val_log_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, best_model = val_loss, epoch, copy.deepcopy(model)
        if epoch - best_epoch >= MLP_PATIENCE:
            break

    if validation is not None:
        model = best_model
    return make_pipeline(scaler, model), fit_time, losses, best_epoch


def train_trees(train, n_estimators, hyperparameters=None):
    defaults = {
        "n_estimators": n_estimators, "min_samples_leaf": 20, "max_features": "sqrt",
        "n_jobs": -1, "random_state": SEED,
    }
    model = ExtraTreesClassifier(**(defaults | (hyperparameters or {})))
    start = perf_counter()
    model.fit(train.x, train.y)
    return model, perf_counter() - start


def point_predictions(batch, probs, folds=None):
    output = batch.rows.copy()
    if folds is not None:
        output["fold"] = folds
    output["prob"] = probs
    return output


def run_cv(name, train, folds, output, n_estimators, hyperparameters=None, save_oof=True):
    stats, loss_rows = [], []
    oof_probs = np.full(len(train.y), np.nan)
    for fold in range(N_FOLDS):
        fold_start = perf_counter()
        is_validation = folds == fold
        fit = train.take(np.flatnonzero(~is_validation))
        validation = train.take(np.flatnonzero(is_validation))
        if name == "mlp":
            fit, chosen_trips = subsample_trips(fit)
            model, train_time, losses, best_epoch = train_mlp(
                fit, validation, hyperparameters=hyperparameters,
            )
        else:
            model, train_time = train_trees(fit, n_estimators, hyperparameters)

        train_probs, _ = predict(model, fit)
        val_probs, infer_time = predict(model, validation)
        oof_probs[is_validation] = val_probs
        report = {
            "fold": fold,
            "n_fit_rows": len(fit.y), "n_validation_rows": len(validation.y),
            "n_fit_trips": fit.rows["trip_id"].nunique(),
            "n_validation_trips": validation.rows["trip_id"].nunique(),
            **validation.evaluate(val_probs),
            **{f"train_{k}": v for k, v in fit.evaluate(train_probs).items()},
            "train_time_s": train_time, "infer_ms_per_1k_rows": infer_time,
            "fold_time_s": perf_counter() - fold_start,
        }
        if name == "mlp":
            report.update(best_epoch=best_epoch, epochs_run=len(losses),
                          fit_trip_ids="|".join(map(str, chosen_trips)))
            loss_rows.extend({"fold": fold, **row} for row in losses)
            pd.DataFrame(loss_rows).to_csv(output / "loss_epochs.csv", index=False)
        else:
            report["n_estimators"] = n_estimators

        stats.append(report)
        pd.DataFrame(stats).to_csv(output / "cv_folds.csv", index=False)
        print(
            f"{name} fold={fold}: top1={report['top1_acc']:.6f} "
            f"macro_f1={report['macro_f1']:.6f} log_loss={report['log_loss']:.6f} "
            f"train={train_time:.1f}s fold={report['fold_time_s']:.1f}s"
            + (f" best_epoch={best_epoch}/{len(losses)}" if name == "mlp" else ""),
            flush=True,
        )
        if (name == "extra_trees" and n_estimators == 200
                and report["fold_time_s"] > TREE_FOLD_BUDGET_S):
            return None

    if not np.isfinite(oof_probs).all():
        raise RuntimeError("A training candidate has no out-of-fold probability.")
    if save_oof:
        point_predictions(train, oof_probs, folds).to_parquet(output / "oof_train.parquet", index=False)
    return pd.DataFrame(stats)


def save_loss_plot(output, cv):
    losses = pd.read_csv(output / "loss_epochs.csv")
    best_epochs = cv.set_index("fold")["best_epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, layout="constrained")
    for ax, (fold, history) in zip(axes.flat, losses.groupby("fold", sort=True)):
        ax.plot(history.epoch, history.train_log_loss, label="Train (25 trips)")
        ax.plot(history.epoch, history.val_log_loss, label="Validation")
        ax.axvline(best_epochs.loc[fold], color="0.4", linestyle="--", label="Best epoch")
        ax.set(title=f"Fold {fold}", xlabel="Epoch", ylabel="Candidate log loss")
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("MLP training and validation log loss")
    fig.savefig(output / "loss_curves.png", dpi=150)
    plt.close(fig)


def save_confusion(output, metrics, name):
    counts = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
    ConfusionMatrixDisplay(np.asarray(counts), display_labels=[0, 1]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set_title(f"{name}: student-test candidate predictions")
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return counts


def run(name, train, student_test, folds, *, hyperparameters=None, output_name=None):
    output = OUTPUT_DIR / (output_name or name)
    output.mkdir(parents=True, exist_ok=True)
    overrides = dict(hyperparameters or {})
    n_estimators = overrides.pop("n_estimators", 200)
    cv = run_cv(name, train, folds, output, n_estimators, overrides)
    retried = cv is None
    if retried:
        (output / "cv_folds.csv").replace(output / "cv_folds_200_tree_attempt.csv")
        n_estimators = 100
        print("A 200-tree fold exceeded 300s. Restarting all five folds at 100 trees.", flush=True)
        cv = run_cv(name, train, folds, output, n_estimators, overrides)

    metric_names = [
        "top1_acc", "macro_f1", "precision_1", "recall_1", "log_loss", "tn", "fp", "fn", "tp",
    ]
    metric_cols = [
        *metric_names, *(f"train_{key}" for key in metric_names),
        "train_time_s", "infer_ms_per_1k_rows",
    ]
    cv_summary = {
        key: {"mean": float(cv[key].mean()), "std": float(cv[key].std(ddof=1))}
        for key in metric_cols
    }
    print(
        f"{name} five-fold CV: top1={cv_summary['top1_acc']['mean']:.6f} "
        f"macro_f1={cv_summary['macro_f1']['mean']:.6f} "
        f"log_loss={cv_summary['log_loss']['mean']:.6f}", flush=True,
    )

    fit = train
    extra = {}
    if name == "mlp":
        fit, chosen_trips = subsample_trips(train)
        refit_epochs = int(cv["best_epoch"].median())
        model, fit_time, _, _ = train_mlp(
            fit, refit_epochs=refit_epochs, hyperparameters=overrides,
        )
        save_loss_plot(output, cv)
        params = model[-1].get_params()
        extra = {
            "mlp_trips_per_fit": MLP_TRIPS, "mlp_max_epochs": MLP_MAX_EPOCHS,
            "mlp_patience": MLP_PATIENCE, "refit_epochs": refit_epochs,
            "refit_trip_ids": list(map(str, chosen_trips)),
        }
    else:
        model, fit_time = train_trees(fit, n_estimators, overrides)
        params = model.get_params()
        extra = {"retried_with_100_trees": retried}
        pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False,
        ).to_csv(output / "feature_importances.csv", index=False)

    probs, infer_time = predict(model, student_test)
    metrics = student_test.evaluate(probs)
    metrics.update(train_time_s=fit_time, infer_ms_per_1k_rows=infer_time)
    metrics["confusion_matrix"] = save_confusion(output, metrics, name)
    point_predictions(student_test, probs).to_parquet(output / "pred_student_test.parquet", index=False)
    model_path = output / "model.joblib"
    temp_model = model_path.with_suffix(".tmp")
    joblib.dump(model, temp_model, compress=3)
    temp_model.replace(model_path)
    summary = {
        "model": name, "feature_cols": FEATURE_COLS, "hyperparameters": params,
        "n_train_rows": len(fit.y), "n_usable_train_rows": len(train.y),
        "n_student_test_rows": len(student_test.y), "cv": cv_summary,
        "student_test": metrics, "model_bytes": model_path.stat().st_size,
        **extra,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{name} student_test: {json.dumps(metrics)}", flush=True)
    print(f"Saved artifacts to {output.resolve()}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["extra_trees", "mlp"])
    args = parser.parse_args()
    train = load_split("train")
    student_test = load_split("student_test")
    if set(train.rows.trip_id) & set(student_test.rows.trip_id):
        raise ValueError("Training and student-test trips overlap.")
    folds = fold_ids(train)
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.model, train, student_test, folds)


if __name__ == "__main__":
    main()
