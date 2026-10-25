"""Run five-fold ensemble prediction without requiring ground-truth labels."""
import argparse
from pathlib import Path

import pandas as pd

from models.common import (CLASSES, ClinicalData, load_ensemble, load_feature,
                           predict_bag, resolve_device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["wsi_only", "wsi_clinical"], required=True)
    parser.add_argument("--features", type=Path, required=True,
                        help="Feature file, directory, or CSV manifest with feature_file and patient_id")
    parser.add_argument("--clinical-csv", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True, help="One mode's config.json and fold_1 through fold_5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if args.output.exists():
        raise FileExistsError("Choose a new output file to preserve previous predictions")
    if args.features.suffix.lower() == ".csv":
        frame = pd.read_csv(args.features, dtype={"patient_id": str})
        if not {"feature_file", "patient_id"}.issubset(frame):
            raise ValueError("Feature manifest requires feature_file and patient_id")
        frame["feature_file"] = frame.feature_file.map(lambda p: str((args.features.resolve().parent / p).resolve()))
    else:
        paths = [args.features] if args.features.is_file() else sorted(args.features.rglob("*.npy"))
        paths = [path for path in paths if not path.stem.endswith("_meta")]
        frame = pd.DataFrame([{"feature_file": str(path.resolve()), "patient_id": path.stem,
                               "label": path.parent.name if path.parent.name in CLASSES else ""} for path in paths])
    if frame.empty:
        raise ValueError("No features found")
    if frame[["feature_file", "patient_id"]].isna().any().any() or frame.patient_id.str.strip().eq("").any():
        raise ValueError("Missing feature path or patient ID")
    if frame.feature_file.duplicated().any():
        raise ValueError("Duplicate feature paths")
    if args.mode == "wsi_clinical" and args.clinical_csv is None:
        raise ValueError("--clinical-csv is required for wsi_clinical")
    clinical = ClinicalData(args.clinical_csv) if args.mode == "wsi_clinical" else None
    device = resolve_device(args.device)
    models, norms, cfg = load_ensemble(args.model_dir, device)
    if cfg["USE_CLINICAL"] != (args.mode == "wsi_clinical"):
        raise ValueError("Selected mode does not match the model configuration")
    rows = []
    for row in frame.itertuples():
        path = Path(row.feature_file)
        scores = predict_bag(load_feature(path), models, norms, device, not args.no_amp,
                             clinical.get(row.patient_id) if clinical else None)
        label = getattr(row, "label", "")
        if pd.isna(label):
            label = ""
        if label and label not in CLASSES:
            raise ValueError(f"Unknown true label: {label}")
        result = {"feature_file": path.name, "patient_id": row.patient_id,
                  "true_class": label, "predicted_class": CLASSES[int(scores.argmax())]}
        result.update({f"prob_{name}": float(scores[i]) for i, name in enumerate(CLASSES)})
        rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predicted {len(rows)} feature files")


if __name__ == "__main__":
    main()
