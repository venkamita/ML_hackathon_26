from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
import pandas as pd

from features.build_candidates import FEATURE_COLS
from models.eval_utils import RANDOM_STATE, inner_validation_trips, load_usable, trip_folds
from models.lgbm import fit_temperature, rank_params, make_ranker, per_point_probabilities
from models.lgbm import train_rank


def three_fold_rank_log_loss(train, params: dict) -> float:
    folds = trip_folds(train["trip_id"], n_splits=3)
    data = train.copy()
    data["fold"] = data["trip_id"].map(folds)
    raw_scores = pd.Series(index=data.index, dtype=float)

    for fold in range(3):
        fit_mask = data["fold"] != fold
        val_mask = data["fold"] == fold
        fit_df = data.loc[fit_mask]
        val_df = data.loc[val_mask]
        inner_fit_trips, inner_val_trips = inner_validation_trips(
            fit_df["trip_id"].unique().tolist()
        )
        inner_fit = fit_df["trip_id"].isin(inner_fit_trips)
        inner_val = fit_df["trip_id"].isin(inner_val_trips)
        model = make_ranker(rank_params(params))
        model.fit(
            fit_df.loc[inner_fit, FEATURE_COLS],
            fit_df.loc[inner_fit, "label"],
            group=fit_df.loc[inner_fit].groupby(["trip_id", "point_idx"], sort=False).size().to_numpy(),
            eval_set=[(fit_df.loc[inner_val, FEATURE_COLS], fit_df.loc[inner_val, "label"])],
            eval_group=[fit_df.loc[inner_val].groupby(["trip_id", "point_idx"], sort=False).size().to_numpy()],
            eval_metric="ndcg",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        raw_scores.loc[val_mask] = model.predict(val_df[FEATURE_COLS])

    temperature = fit_temperature(raw_scores.to_numpy(), data)
    probs = per_point_probabilities(data, raw_scores.to_numpy(), temperature=temperature,
                                    real_valued=True)
    from sklearn.metrics import log_loss
    import numpy as np
    return float(log_loss(data["label"], np.clip(probs, 1e-7, 1 - 1e-7)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    _, train = load_usable("outputs/features/features_train.parquet")
    untuned_path = Path("outputs/models/lgbm_rank/summary.json")
    if not untuned_path.exists():
        raise FileNotFoundError("Run python -m models.lgbm --variant rank first")
    untuned = json.loads(untuned_path.read_text(encoding="utf-8"))

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": 2000,
        }
        return three_fold_rank_log_loss(train, params)

    study.optimize(objective, n_trials=args.trials)
    trials = study.trials_dataframe()
    trials.to_csv("outputs/models/optuna_trials.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax)
    fig.tight_layout()
    fig.savefig("outputs/models/optuna_optimization_history.png", dpi=150)
    plt.close(fig)

    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        optuna.visualization.matplotlib.plot_param_importances(study, ax=ax)
        fig.tight_layout()
        fig.savefig("outputs/models/optuna_parameter_importance.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Parameter-importance plot unavailable: {exc}")

    best_params = dict(study.best_params)
    best_params["n_estimators"] = 2000
    _, test = load_usable("outputs/features/features_student_test.parquet")
    tuned = train_rank(train, test, extra_params=best_params,
                       output_name="lgbm_rank_tuned")

    comparison = pd.DataFrame([
        {
            "variant": "untuned",
            "cv_macro_f1": untuned["macro_f1_mean"],
            "student_top1": untuned["student_test_metrics"]["top1_acc"],
            "student_macro_f1": untuned["student_test_metrics"]["macro_f1"],
            "student_log_loss": untuned["student_test_metrics"]["log_loss"],
            "student_infer_ms_per_1k_rows": untuned["student_test_infer_ms_per_1k_rows"],
        },
        {
            "variant": "tuned",
            "cv_macro_f1": tuned["macro_f1_mean"],
            "student_top1": tuned["student_test_metrics"]["top1_acc"],
            "student_macro_f1": tuned["student_test_metrics"]["macro_f1"],
            "student_log_loss": tuned["student_test_metrics"]["log_loss"],
            "student_infer_ms_per_1k_rows": tuned["student_test_infer_ms_per_1k_rows"],
        },
    ])
    comparison.to_csv("outputs/models/tuned_vs_untuned.csv", index=False)
    (Path("outputs/models/optuna_summary.json")).write_text(
        json.dumps({
            "n_trials": args.trials,
            "best_trial": study.best_trial.number,
            "best_3fold_cv_log_loss": study.best_value,
            "best_params": best_params,
            "comparison": comparison.to_dict(orient="records"),
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"best_params": best_params, "best_value": study.best_value}, indent=2))


if __name__ == "__main__":
    main()
