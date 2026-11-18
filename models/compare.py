from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.build_candidates import FEATURE_COLS


def main() -> None:
    model_root = Path("outputs/models")
    summaries = []
    for path in sorted(model_root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append({
            "model": summary.get("model_name", path.parent.name),
            "cv_top1_mean": summary["top1_acc_mean"],
            "cv_top1_std": summary["top1_acc_std"],
            "cv_log_loss_mean": summary["log_loss_mean"],
            "cv_log_loss_std": summary["log_loss_std"],
            "cv_macro_f1_mean": summary["macro_f1_mean"],
            "cv_macro_f1_std": summary["macro_f1_std"],
            "student_top1": summary["student_test_metrics"]["top1_acc"],
            "student_macro_f1": summary["student_test_metrics"]["macro_f1"],
            "student_log_loss": summary["student_test_metrics"]["log_loss"],
            "infer_ms_per_1k_rows": summary["infer_ms_per_1k_rows_mean"]
            if "infer_ms_per_1k_rows_mean" in summary
            else summary.get("student_test_infer_ms_per_1k_rows", np.nan),
        })
    if not summaries:
        raise RuntimeError("No outputs/models/<name>/summary.json files found")

    table = pd.DataFrame(summaries)
    table.to_csv(model_root / "comparison.csv", index=False)

    def bar(metric, std, title, filename):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(table["model"], table[metric], yerr=table[std], capsize=4)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(model_root / filename, dpi=150)
        plt.close(fig)

    bar("cv_top1_mean", "cv_top1_std", "CV Top-1 accuracy",
        "comparison_cv_top1.png")
    bar("cv_log_loss_mean", "cv_log_loss_std", "CV row log loss",
        "comparison_cv_log_loss.png")

    ranked = table.sort_values(
        ["cv_top1_mean", "cv_log_loss_mean", "cv_top1_std", "infer_ms_per_1k_rows"],
        ascending=[False, True, True, True],
    )
    chosen = ranked.iloc[0]
    source_dir = model_root / str(chosen["model"])
    shutil.copy2(source_dir / "model.joblib", model_root / "final_model.joblib")
    (model_root / "final_features.json").write_text(
        json.dumps(FEATURE_COLS, indent=2), encoding="utf-8"
    )
    (model_root / "final_model_justification.txt").write_text(
        f"{chosen['model']} was selected by the required ranking rule: highest CV Top-1 accuracy "
        f"({chosen['cv_top1_mean']:.4f}), with CV log loss used as the first tie-breaker "
        f"({chosen['cv_log_loss_mean']:.4f}). Fold stability and inference speed were considered "
        f"next, with {chosen['cv_top1_std']:.4f} Top-1 fold standard deviation and "
        f"{chosen['infer_ms_per_1k_rows']:.3f} ms per 1k rows. This choice prioritizes the current "
        "Top-1 metric while retaining log loss as the probability-quality tie-breaker and checking "
        "generalisation and deployment cost.",
        encoding="utf-8",
    )
    print(ranked.to_string(index=False))
    print(f"Final model: {chosen['model']}")


if __name__ == "__main__":
    main()
