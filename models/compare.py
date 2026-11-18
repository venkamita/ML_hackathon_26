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
        cv_s = summary.get("cv_summary", {})
        cv_d = summary.get("cv", {})
        st_m = summary.get("student_test_metrics", summary.get("student_test", {}))

        def _get_val(k):
            if k in summary:
                return summary[k]
            if k in cv_s:
                return cv_s[k]
            clean_k = k.replace("_mean", "")
            if clean_k in cv_d and isinstance(cv_d[clean_k], dict):
                return cv_d[clean_k].get("mean", np.nan)
            return np.nan

        def _get_std(k):
            std_k = k.replace("_mean", "_std")
            if std_k in summary:
                return summary[std_k]
            if std_k in cv_s:
                return cv_s[std_k]
            clean_k = k.replace("_mean", "")
            if clean_k in cv_d and isinstance(cv_d[clean_k], dict):
                return cv_d[clean_k].get("std", 0.0)
            return 0.0

        infer_speed = summary.get("infer_ms_per_1k_rows_mean",
            cv_s.get("infer_ms_per_1k_rows_mean",
            cv_d.get("infer_ms_per_1k_rows", {}).get("mean",
            st_m.get("infer_ms_per_1k_rows", np.nan))))

        summaries.append({
            "model": summary.get("model_name", summary.get("model", path.parent.name)),
            "cv_top1_mean": _get_val("top1_acc_mean"),
            "cv_top1_std": _get_std("top1_acc_mean"),
            "cv_log_loss_mean": _get_val("log_loss_mean"),
            "cv_log_loss_std": _get_std("log_loss_std"),
            "cv_macro_f1_mean": _get_val("macro_f1_mean"),
            "cv_macro_f1_std": _get_std("macro_f1_std"),
            "student_top1": st_m.get("top1_acc", summary.get("student_test_top1_acc", np.nan)),
            "student_macro_f1": st_m.get("macro_f1", summary.get("student_test_macro_f1", np.nan)),
            "student_log_loss": st_m.get("log_loss", summary.get("student_test_log_loss", np.nan)),
            "infer_ms_per_1k_rows": infer_speed,
        })
    if not summaries:
        raise RuntimeError("No outputs/models/<name>/summary.json files found")

    table = pd.DataFrame(summaries)
    table.to_csv(model_root / "comparison.csv", index=False)

    def bar(metric, std, title, filename):
        fig, ax = plt.subplots(figsize=(9, 5))
        yerr = table[std] if std in table.columns and not table[std].isna().all() else None
        ax.bar(table["model"], table[metric], yerr=yerr, capsize=4, color="#1f77b4", alpha=0.85)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(model_root / filename, dpi=150)
        plt.close(fig)

    bar("cv_top1_mean", "cv_top1_std", "CV Top-1 Accuracy", "comparison_cv_top1.png")
    bar("cv_log_loss_mean", "cv_log_loss_std", "CV Row Log Loss", "comparison_cv_log_loss.png")
    bar("cv_macro_f1_mean", "cv_macro_f1_std", "CV Macro F1-score", "comparison_cv_macro_f1.png")
    bar("student_top1", "cv_top1_std", "Student-Test Top-1 Accuracy", "comparison_student_top1.png")
    bar("student_macro_f1", "cv_macro_f1_std", "Student-Test Macro F1-score", "comparison_student_macro_f1.png")

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
