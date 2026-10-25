"""Five-fold training with explicit patient splits and compatible checkpoints."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.common import (CLASSES, ClinicalData, build_model, fit_normalizer,
                           load_feature, normalize_clinical, resolve_device)


def validate_manifest(frame, n_folds=5, check_files=True):
    required = {"feature_file", "patient_id", "label", "fold", "variant"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Nonempty manifest requires {sorted(required)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Manifest fields cannot be missing")
    if frame.patient_id.astype(str).str.strip().eq("").any():
        raise ValueError("Empty patient identifier")
    if frame.feature_file.duplicated().any():
        raise ValueError("Feature files must be unique")
    if not frame.label.isin(CLASSES).all() or not frame.variant.isin(["original", "augmented"]).all():
        raise ValueError("Unknown label or variant")
    folds = pd.to_numeric(frame.fold, errors="raise")
    if not ((folds % 1 == 0) & folds.between(0, n_folds)).all():
        raise ValueError(f"fold must be an integer from 0 to {n_folds}")
    for _, group in frame.groupby("patient_id"):
        if group.fold.nunique() != 1 or group.label.nunique() != 1:
            raise ValueError("Each patient must have one fold and one label")
    if not frame.loc[folds == 0, "variant"].eq("original").all():
        raise ValueError("Hold-out evaluation accepts original features only")
    for fold in range(1, n_folds + 1):
        if not ((folds == fold) & frame.variant.eq("original")).any():
            raise ValueError(f"Missing original validation samples for fold {fold}")
    if check_files:
        missing = [str(path) for path in frame.feature_file if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing feature files: {missing[:3]}")


def read_manifest(path, check_files=True):
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype={"patient_id": str, "feature_file": str})
    if "feature_file" in frame:
        frame["feature_file"] = frame.feature_file.map(
            lambda value: str((path.parent / value).resolve()) if isinstance(value, str) else value)
    validate_manifest(frame, check_files=check_files)
    frame["fold"] = frame.fold.astype(int)
    return frame


class BagDataset(Dataset):
    def __init__(self, frame, clinical=None, normalizer=None):
        self.frame = frame.reset_index(drop=True)
        self.clinical = clinical
        self.normalizer = normalizer

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        x = load_feature(row.feature_file).half()
        values, mask = np.zeros(15, np.float32), np.zeros(15, np.float32)
        if self.clinical:
            values, mask = self.clinical.get(row.patient_id)
            values = normalize_clinical(values, mask, self.normalizer)
        return x, torch.from_numpy(values), torch.from_numpy(mask), CLASSES.index(row.label)


def collate_bags(batch):
    xs, values, masks, labels = zip(*batch)
    if len({tuple(x.shape) for x in xs}) != 1:
        raise ValueError("Batched training requires equal patch counts; use --batch-size 1 for variable counts")
    return torch.stack(xs), torch.stack(values), torch.stack(masks), torch.tensor(labels)


def batch_loss(model, batch, device, cfg):
    x, values, masks, labels = [value.to(device, non_blocking=True) for value in batch]
    if not cfg["USE_AMP"]:
        x = x.float()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg["USE_AMP"]):
        logits = model(x, clin=values, clin_mask=masks) if cfg["USE_CLINICAL"] else model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=cfg["LABEL_SMOOTHING"])
    return loss


def train_fold(frame, fold, cfg, output, device, clinical=None):
    train = frame[(frame.fold != 0) & (frame.fold != fold)]
    valid = frame[(frame.fold == fold) & frame.variant.eq("original")]
    if train.empty or valid.empty:
        raise ValueError("Empty training or validation partition")
    folder = output / f"fold_{fold}"
    if folder.exists():
        raise FileExistsError(f"Refusing to replace existing fold directory: {folder}")
    folder.mkdir(parents=True)
    normalizer = fit_normalizer(clinical, train.patient_id) if clinical else None
    if normalizer:
        (folder / "clinical_norm.json").write_text(json.dumps(normalizer, indent=2), encoding="utf-8")
    # Split manifests stay local with the generated checkpoint bundle.
    train.to_csv(folder / "train_manifest.csv", index=False)
    valid.to_csv(folder / "validation_manifest.csv", index=False)
    loaders = []
    for part, shuffle in ((train, True), (valid, False)):
        kwargs = dict(dataset=BagDataset(part, clinical, normalizer), batch_size=cfg["BATCH_SIZE"],
                      shuffle=shuffle, num_workers=cfg["NUM_WORKERS"], pin_memory=True,
                      collate_fn=collate_bags)
        if cfg["NUM_WORKERS"] > 0:
            kwargs.update(persistent_workers=cfg["PERSISTENT_WORKERS"], prefetch_factor=cfg["PREFETCH_FACTOR"])
        loaders.append(DataLoader(**kwargs))
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["USE_AMP"])
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()
        total = 0.
        for batch in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, device, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP_NORM"])
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(batch[-1])
        model.eval()
        with torch.no_grad():
            val = sum(batch_loss(model, batch, device, cfg).item() * len(batch[-1]) for batch in loaders[1]) / len(valid)
        if not np.isfinite([total, val]).all():
            raise RuntimeError("Non-finite training or validation loss")
        history.append({"epoch": epoch, "training_loss": total / len(train), "validation_loss": val})
        print(f"Fold {fold}, epoch {epoch}: validation loss {val:.6f}")
        if val < best - cfg["ES_MIN_DELTA"]:
            best, stale = val, 0
            torch.save({"model_state": model.state_dict(), "class_names": CLASSES,
                        "cfg": cfg, "fold": fold, "epoch": epoch}, folder / "best.pt")
        else:
            stale += 1
            if cfg["EARLY_STOP"] and stale >= cfg["ES_PATIENCE"]:
                break
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clinical-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for argument, key in ((args.epochs, "EPOCHS"), (args.batch_size, "BATCH_SIZE"), (args.workers, "NUM_WORKERS")):
        if argument is not None:
            cfg[key] = argument
    if cfg["N_FOLDS"] != 5 or cfg["EPOCHS"] < 1 or cfg["BATCH_SIZE"] < 1 or cfg["NUM_WORKERS"] < 0:
        raise ValueError("Expected five folds, positive epochs/batch size and nonnegative workers")
    cfg["USE_AMP"] = cfg["USE_AMP"] and not args.no_amp
    frame = read_manifest(args.manifest)
    if cfg["USE_CLINICAL"] and args.clinical_csv is None:
        raise ValueError("--clinical-csv is required for clinical training")
    clinical = ClinicalData(args.clinical_csv) if cfg["USE_CLINICAL"] else None
    for row in frame.itertuples():
        load_feature(row.feature_file)
        if clinical:
            clinical.get(row.patient_id)
    if args.validate_only:
        print("Input validation passed")
        return
    device = resolve_device(args.device)
    random.seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])
    torch.manual_seed(cfg["SEED"])
    torch.cuda.manual_seed_all(cfg["SEED"])
    torch.backends.cuda.matmul.allow_tf32 = cfg["USE_TF32"]
    torch.backends.cudnn.allow_tf32 = cfg["USE_TF32"]
    torch.backends.cudnn.benchmark = cfg["CUDNN_BENCHMARK"]
    torch.set_float32_matmul_precision(cfg["MATMUL_PRECISION"])
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve existing training runs")
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for fold in range(1, 6):
        train_fold(frame, fold, cfg, args.output, device, clinical)


if __name__ == "__main__":
    main()
