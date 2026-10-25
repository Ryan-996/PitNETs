"""Evaluate labeled predictions and export metrics, plots, and paired comparisons.

Overall Accuracy uses correct class decisions; other Overall metrics use five-class macro averages.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

from models.common import CLASSES

METRICS = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1-score", "AUC"]


def validate_scores(labels, scores):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=str)
    if not len(labels) or scores.shape != (len(labels), 5) or not set(labels).issubset(CLASSES):
        raise ValueError("Expected known labels and a nonempty (N, 5) probability matrix")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not np.allclose(scores.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError("Probability rows must sum to one")
    truth = np.asarray([CLASSES.index(label) for label in labels])
    return truth, scores


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(actual, called):
    tp = int((actual & called).sum())
    tn = int((~actual & ~called).sum())
    fp = int((~actual & called).sum())
    fn = int((actual & ~called).sum())
    return {"Accuracy": safe_ratio(tp + tn, len(actual)),
            "Sensitivity": safe_ratio(tp, tp + fn), "Specificity": safe_ratio(tn, tn + fp),
            "PPV": safe_ratio(tp, tp + fp), "NPV": safe_ratio(tn, tn + fn),
            "F1-score": safe_ratio(2 * tp, 2 * tp + fp + fn)}


def overall_auc(truth, scores, cohort):
    """Return the five-class macro OvR AUC for either cohort."""
    if len(np.unique(truth)) != 5:
        return float("nan")
    return float(np.mean([roc_auc_score(truth == i, scores[:, i]) for i in range(5)]))


def calculate_metrics(labels, scores, cohort):
    if cohort not in ("internal", "external"):
        raise ValueError("Cohort must be internal or external")
    truth, scores = validate_scores(labels, scores)
    predicted = scores.argmax(1)
    rows = []
    for i, name in enumerate(CLASSES):
        actual = truth == i
        value = float(roc_auc_score(actual, scores[:, i])) if 0 < actual.sum() < len(truth) else float("nan")
        rows.append({"Class": name, **binary_metrics(actual, predicted == i), "AUC": value})
    overall = {"Class": "Overall", "Accuracy": float((predicted == truth).mean()),
               **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS[1:]}}
    return pd.DataFrame([overall, *rows], columns=["Class", *METRICS])


def read_predictions(path):
    frame = pd.read_csv(path, dtype={"patient_id": str, "feature_file": str})
    required = ["patient_id", "feature_file", "true_class", *[f"prob_{name}" for name in CLASSES]]
    if not set(required).issubset(frame) or frame[required].isna().any().any():
        raise ValueError("Labeled prediction CSV requires patient_id, feature_file, true_class and five probability columns")
    if frame.patient_id.str.strip().eq("").any() or frame.feature_file.str.strip().eq("").any():
        raise ValueError("Empty patient or feature identifier")
    if frame.duplicated(["patient_id", "feature_file"]).any():
        raise ValueError("Duplicate patient/feature identifier")
    if (frame.groupby("patient_id").true_class.nunique() > 1).any():
        raise ValueError("Each patient must have one true class")
    validate_scores(frame.true_class, frame[[f"prob_{name}" for name in CLASSES]])
    return frame


def plot_results(labels, scores, output, cohort):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    truth, scores = validate_scores(labels, scores)
    fig, axis = plt.subplots(figsize=(6, 5))
    matrix = confusion_matrix(truth, scores.argmax(1), labels=range(5))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES,
             yticklabels=CLASSES, xlabel="Predicted class", ylabel="True class")
    for i in range(5):
        for j in range(5):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(output / "confusion_matrix.csv")
    fig, axis = plt.subplots(figsize=(6, 5))
    curves = []
    for i, name in enumerate(CLASSES):
        actual = truth == i
        if 0 < actual.sum() < len(truth):
            fpr, tpr, _ = roc_curve(actual, scores[:, i])
            curves.append((fpr, tpr))
            axis.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    if len(curves) == 5:
        grid = np.unique(np.concatenate([fpr for fpr, _ in curves]))
        mean_tpr = np.mean([np.interp(grid, fpr, tpr) for fpr, tpr in curves], axis=0)
        axis.plot(grid, mean_tpr, linewidth=2, label=f"Overall (AUC = {overall_auc(truth, scores, cohort):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "roc_curves.png", dpi=200)
    plt.close(fig)


def paired_comparison(left, right, cohort, iterations=10000, seed=42):
    """Align common observations; resample and swap whole patients."""
    paired = left.merge(right, on=["patient_id", "feature_file"], suffixes=("_a", "_b"), validate="one_to_one")
    if paired.empty or not paired.true_class_a.equals(paired.true_class_b):
        raise ValueError("Paired predictions require shared identifiers and identical labels")
    labels = paired.true_class_a.to_numpy()
    truth = np.asarray([CLASSES.index(label) for label in labels])
    a = paired[[f"prob_{name}_a" for name in CLASSES]].to_numpy(float)
    b = paired[[f"prob_{name}_b" for name in CLASSES]].to_numpy(float)
    table_a, table_b = calculate_metrics(labels, a, cohort), calculate_metrics(labels, b, cohort)
    groups = [np.asarray(indices) for indices in paired.groupby("patient_id", sort=True).indices.values()]
    strata = {}
    for group in groups:
        strata.setdefault(truth[group[0]], []).append(group)
    rng = np.random.default_rng(seed)
    difference = float(table_b.iloc[0]["AUC"] - table_a.iloc[0]["AUC"])
    boot, extreme = [], 0
    if np.isfinite(difference):
        for _ in range(iterations):
            indices = np.concatenate([stratum[i] for stratum in strata.values()
                                      for i in rng.integers(0, len(stratum), size=len(stratum))])
            boot.append(overall_auc(truth[indices], b[indices], cohort) - overall_auc(truth[indices], a[indices], cohort))
            swap = np.zeros(len(paired), dtype=bool)
            for group in groups:
                swap[group] = bool(rng.integers(0, 2))
            aa, bb = np.where(swap[:, None], b, a), np.where(swap[:, None], a, b)
            value = overall_auc(truth, bb, cohort) - overall_auc(truth, aa, cohort)
            extreme += abs(value) >= abs(difference) - 1e-12
    low, high = np.quantile(boot, [.025, .975]) if boot else (float("nan"), float("nan"))
    correct_a, correct_b = a.argmax(1) == truth, b.argmax(1) == truth
    ab, ba = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
    repeated = len(groups) != len(paired)
    return pd.DataFrame([
        {"Metric": "AUC", "Model A": table_a.iloc[0]["AUC"], "Model B": table_b.iloc[0]["AUC"],
         "Difference": difference, "Lower": low, "Upper": high,
         "P-value": (extreme + 1) / (iterations + 1) if boot else np.nan,
         "Method": "Paired patient permutation; patient bootstrap interval"},
        {"Metric": "Accuracy", "Model A": correct_a.mean(), "Model B": correct_b.mean(),
         "Difference": correct_b.mean() - correct_a.mean(), "Lower": np.nan, "Upper": np.nan,
         "P-value": np.nan if repeated else (binomtest(min(ab, ba), ab + ba, .5).pvalue if ab + ba else 1.),
         "Method": "Unavailable: repeated patient observations" if repeated else "Exact McNemar; correct class decisions"},
    ]).assign(Observations=len(paired), Patients=len(groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--cohort", choices=["internal", "external"], required=True,
                        help="Cohort label recorded in notes; metric definitions are identical")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("Iterations must be positive")
    frame = read_predictions(args.predictions)
    scores = frame[[f"prob_{name}" for name in CLASSES]].to_numpy(float)
    comparison = paired_comparison(frame, read_predictions(args.compare), args.cohort,
                                   args.iterations, args.seed) if args.compare else None
    if args.output.exists():
        raise FileExistsError("Choose a new evaluation output directory")
    args.output.mkdir(parents=True)
    table = calculate_metrics(frame.true_class, scores, args.cohort)
    table.to_csv(args.output / "metrics.csv", index=False, na_rep="NA")
    plot_results(frame.true_class, scores, args.output, args.cohort)
    if comparison is not None:
        comparison.to_csv(args.output / "paired_comparison.csv", index=False, na_rep="NA")
    notes = [f"Cohort: {args.cohort}", f"Observations: {len(frame)}", f"Patients: {frame.patient_id.nunique()}",
             "Point estimates use feature-file observations. Repeated slides are not independent patients.",
             "Overall Accuracy: fraction of correctly classified feature-file observations.",
             "Other Overall metrics: unweighted five-class means; AUC uses one-vs-rest.",
             "Undefined class metrics propagate to the corresponding Overall macro metric as NA; no classes are omitted.",
             "Metric definitions are identical for internal and external cohorts.",
             "Earlier evaluations used one-hot element Overall Accuracy, pooled non-AUC Overall metrics, and internal micro AUC."]
    if comparison is not None:
        notes.append("Paired comparisons use common patient_id/feature_file observations; bootstrap and permutation operate on whole patients.")
    (args.output / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(table.to_string(index=False, na_rep="NA"))


if __name__ == "__main__":
    main()
