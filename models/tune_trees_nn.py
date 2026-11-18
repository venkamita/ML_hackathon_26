"""Select and tune one PathSnap model using only grouped training CV.

Install Optuna and run both baselines before running from the repository root:
    python -m pip install optuna==4.6.0
    python -m models.tune_trees_nn --model best --trials 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
from time import perf_counter

import optuna
import pandas as pd
from threadpoolctl import threadpool_limits

from models import trees_nn as baseline

MODELS = ("extra_trees", "mlp")
OBJECTIVE = "top1_acc"
STARTUP_TRIALS = 4
SEARCH_SPACES = {
    "extra_trees": {
        "n_estimators": [100, 150, 200],
        "min_samples_leaf": [8, 12, 20, 30, 40],
        "max_features": ["sqrt", 0.25, 0.4],
    },
    "mlp": {
        "hidden_layer_sizes": ["32x16", "64x32", "64x64", "96x48"],
        "alpha": {"low": 1e-6, "high": 1e-2, "log": True},
        "batch_size": [512, 1024, 2048],
        "learning_rate_init": {"low": 3e-4, "high": 3e-3, "log": True},
    },
}
SCORE_KEYS = ("top1_acc", "macro_f1", "log_loss", "precision_1", "recall_1")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def get_baseline(name):
    path = baseline.OUTPUT_DIR / name / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def effective_params(params):
    config = dict(params)
    if "hidden_layer_sizes" in config:
        config["hidden_layer_sizes"] = tuple(map(int, config["hidden_layer_sizes"].split("x")))
    return config


def tune(name, train, folds, base, n_trials):
    output = baseline.OUTPUT_DIR / f"{name}_tuned"
    output.mkdir(parents=True, exist_ok=True)
    space = SEARCH_SPACES[name]
    with (baseline.FEATURE_DIR / "features_train.parquet").open("rb") as stream:
        train_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config = {
        "model": name, "objective": f"maximize mean five-fold {OBJECTIVE}",
        "sampler": "Optuna TPESampler", "optuna_version": optuna.__version__,
        "startup_trials": STARTUP_TRIALS, "seed": baseline.SEED, "requested_total_trials": n_trials,
        "fold_assignment": "sorted usable trip IDs, zero-based position modulo 5",
        "train_feature_sha256": train_hash, "search_space": space,
    }
    write_json(output / "search_config.json", config)
    fold_map = train.rows[["trip_id"]].assign(fold=folds).drop_duplicates()
    fold_map.to_csv(output / "trip_folds.csv", index=False)

    sampler = optuna.samplers.TPESampler(seed=baseline.SEED, n_startup_trials=STARTUP_TRIALS)
    study = optuna.create_study(
        study_name=f"{name}-grouped-top1", sampler=sampler, direction="maximize",
        storage=f"sqlite:///{(output / 'study.sqlite3').as_posix()}",
        load_if_exists=True,
    )
    if not study.trials:
        study.set_user_attr("training_feature_sha256", train_hash)
        study.set_user_attr("search_space", space)
        initial = {key: base["hyperparameters"][key] for key in space}
        if name == "mlp":
            initial["hidden_layer_sizes"] = "x".join(map(str, initial["hidden_layer_sizes"]))
        study.enqueue_trial(initial, user_attrs={"baseline_configuration": True})
    elif (study.user_attrs.get("training_feature_sha256") != train_hash
          or study.user_attrs.get("search_space") != space):
        raise ValueError("The stored study belongs to different data or a different search space.")

    def objective(trial):
        picked = {}
        for key, domain in space.items():
            if isinstance(domain, list):
                picked[key] = trial.suggest_categorical(key, domain)
            else:
                picked[key] = trial.suggest_float(key, **domain)
        overrides = effective_params(picked)
        n_estimators = overrides.pop("n_estimators", 200)
        trial_dir = output / "trials" / f"{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(trial_dir / "params.json", picked)
        print(f"trial={trial.number} params={picked}", flush=True)
        start = perf_counter()
        try:
            cv = baseline.run_cv(name, train, folds, trial_dir, n_estimators, overrides, save_oof=False)
        finally:
            trial.set_user_attr("cv_wall_time_s", perf_counter() - start)
        if cv is None:
            trial.set_user_attr("stop_reason", "200-tree fold exceeded 300 seconds")
            raise optuna.TrialPruned("200-tree fold exceeded 300 seconds.")
        for key in (*SCORE_KEYS, "train_time_s", "infer_ms_per_1k_rows"):
            trial.set_user_attr(f"cv_mean_{key}", float(cv[key].mean()))
            trial.set_user_attr(f"cv_std_{key}", float(cv[key].std(ddof=1)))
        return float(cv[OBJECTIVE].mean())

    def log_trial(study, trial):
        study.trials_dataframe().to_csv(output / "trials.csv", index=False)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"trial={trial.number} mean_top1={trial.value:.8f}; "
                  f"best_trial={study.best_trial.number} best_top1={study.best_value:.8f}", flush=True)
        else:
            print(f"trial={trial.number} status={trial.state.name}", flush=True)

    remaining = max(0, n_trials - sum(trial.state.is_finished() for trial in study.trials))
    study.optimize(objective, n_trials=remaining, callbacks=[log_trial])
    write_json(output / "best_trial.json", {
        "trial": study.best_trial.number, "mean_cv_top1_acc": study.best_value,
        "parameters": study.best_params, "completed_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        ), "trial_count": len(study.trials),
    })
    return output, study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("best", *MODELS), default="best")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if args.trials < STARTUP_TRIALS + 1:
        parser.error(f"Use at least {STARTUP_TRIALS + 1} trials to include adaptive TPE trials.")

    if args.model == "best":
        summaries = {name: get_baseline(name) for name in MODELS}
        name = max(summaries, key=lambda model: summaries[model]["cv"][OBJECTIVE]["mean"])
        base = summaries[name]
    else:
        name = args.model
        base = get_baseline(name)
    print(f"Selected {name} using baseline mean CV {OBJECTIVE}={base['cv'][OBJECTIVE]['mean']:.8f}.",
          flush=True)

    train = baseline.load_split("train")
    folds = baseline.fold_ids(train)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        output, study = tune(name, train, folds, base, args.trials)
        search_time = perf_counter() - start
        test = baseline.load_split("student_test")
        if set(train.rows.trip_id) & set(test.rows.trip_id):
            raise ValueError("Training and student-test trips overlap.")
        tuned = baseline.run(
            name, train, test, folds, hyperparameters=effective_params(study.best_params),
            output_name=output.name,
        )

    fields = (*SCORE_KEYS, "train_time_s", "infer_ms_per_1k_rows")
    comparison = []
    for split in ("cv", "student_test"):
        for key in fields:
            before, after = base[split][key], tuned[split][key]
            if split == "cv":
                row = {"split": split, "metric": key, "baseline": before["mean"],
                       "tuned": after["mean"], "baseline_std": before["std"],
                       "tuned_std": after["std"]}
            else:
                row = {"split": split, "metric": key, "baseline": before, "tuned": after}
            row["tuned_minus_baseline"] = row["tuned"] - row["baseline"]
            comparison.append(row)
    pd.DataFrame(comparison).to_csv(output / "baseline_vs_tuned.csv", index=False)
    write_json(output / "search_timing.json", {
        "invocation_search_wall_time_s": search_time,
        "cumulative_trial_wall_time_s": sum(
            trial.user_attrs.get("cv_wall_time_s", 0) for trial in study.trials
        ),
        "total_invocation_wall_time_s": perf_counter() - start,
    })
    print(f"Best trial {study.best_trial.number}: {study.best_params}", flush=True)
    print(f"Saved study and paired comparison to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
