"""Shared release contracts for training and inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .pitnets import PitNETsMIL

CLASSES = ["Multi", "PIT1", "SF1", "TPIT", "zero"]
CLINICAL_COLUMNS = [
    "sex", "age", "visual_symptoms", "headache", "menstrual_disorder",
    "acromegaly", "physical_examination", "other_symptoms", "prolactin",
    "growth_hormone", "acth", "fsh_lh", "thyroid_hormone",
    "no_hormone_elevation", "tumor_diameter",
]


def resolve_device(name="cuda"):
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; select an available GPU with --device cuda:0")
    torch.empty(1, device=device)
    return device


def load_feature(path):
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 1280 or not len(array):
        raise ValueError(f"Expected nonempty (N, 1280) features: {path}; got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"Features must be finite numeric values: {path}")
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))


class ClinicalData:
    """Read English clinical columns by explicit patient ID, without fitting."""

    def __init__(self, path):
        frame = pd.read_csv(path, dtype={"patient_id": str})
        if list(frame.columns) != ["patient_id", *CLINICAL_COLUMNS]:
            raise ValueError("Clinical columns must match the documented English template and order")
        if frame.patient_id.isna().any() or frame.patient_id.str.strip().eq("").any():
            raise ValueError("Clinical patient IDs must be nonempty")
        frame["patient_id"] = frame.patient_id.str.strip()
        if frame.patient_id.duplicated().any():
            raise ValueError("Clinical patient IDs must be unique")
        self.rows = {}
        for _, row in frame.iterrows():
            values = np.zeros(15, dtype=np.float32)
            mask = np.zeros(15, dtype=np.float32)
            for i, column in enumerate(CLINICAL_COLUMNS):
                raw = row[column]
                if pd.isna(raw) or str(raw).strip() == "":
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Non-numeric clinical value in {column}") from error
                if not np.isfinite(value) or (i not in (1, 14) and value not in (0, 1)):
                    raise ValueError(f"Invalid clinical value in {column}")
                values[i], mask[i] = value, 1
            self.rows[row.patient_id] = (values, mask)

    def get(self, patient_id):
        if str(patient_id) not in self.rows:
            raise ValueError(f"No clinical match for patient {patient_id!r}")
        return tuple(value.copy() for value in self.rows[str(patient_id)])


def fit_normalizer(clinical, patient_ids):
    records = [clinical.get(pid) for pid in sorted(set(patient_ids))]
    cont = {}
    for index in (1, 14):
        observed = [values[index] for values, mask in records if mask[index] == 1]
        values = np.asarray(observed, dtype=np.float32)
        mean, std = (float(values.mean()), float(values.std())) if len(values) else (0., 1.)
        cont[str(index)] = {"mean": mean, "std": std if std >= 1e-6 else 1.}
    return {"params": {"cont": cont}}


def normalize_clinical(values, mask, normalizer):
    result = values.copy()
    for index, stats in normalizer["params"]["cont"].items():
        i = int(index)
        if i not in (1, 14) or not np.isfinite([stats["mean"], stats["std"]]).all() or stats["std"] <= 0:
            raise ValueError("Invalid clinical normalizer")
        if mask[i] >= .5:
            result[i] = (result[i] - float(stats["mean"])) / max(float(stats["std"]), 1e-6)
    if set(normalizer["params"]["cont"]) != {"1", "14"}:
        raise ValueError("Clinical normalizer must contain age and tumor diameter")
    return result


def model_kwargs(cfg):
    keys = {"EMBED_DIM": "embed_dim", "DROPOUT": "dropout",
            "ATTN_DIM": "attn_dim", "ATTN_DROPOUT": "attn_dropout",
            "INSTANCE_DROPOUT": "instance_dropout", "CLS_HIDDEN_DIM": "cls_hidden_dim",
            "CLS_NUM_LAYERS": "cls_layers"}
    if cfg["USE_CLINICAL"]:
        keys.update({"CLIN_OUT": "clin_out", "CLIN_DIM": "clin_dim",
                     "GATING_HIDDEN": "gating_hidden", "GATING_DROPOUT": "gating_dropout",
                     "CLIN_PART_HID": "clin_part_hid", "CLIN_PART_OUT": "clin_part_out",
                     "CLIN_ENC_DROPOUT": "clin_enc_dropout", "CLIN_N_HEADS": "clin_n_heads",
                     "CLIN_ATTN_FF": "clin_attn_ff", "CLIN_NUM_LAYERS": "clin_num_layers",
                     "CLIN_FEATURE_MASK_DROP": "clinical_feature_mask_drop",
                     "CLIN_PART_MASK_DROP": "clinical_part_mask_drop"})
    kwargs = {target: cfg[source] for source, target in keys.items()}
    kwargs["in_dim"] = cfg.get("IN_DIM", 1280)
    if cfg.get("PART_SLICES") is not None:
        kwargs["part_slices"] = {key: slice(*value) for key, value in cfg["PART_SLICES"].items()}
    return kwargs


def build_model(cfg):
    return PitNETsMIL(n_classes=5, use_clinical=cfg["USE_CLINICAL"], **model_kwargs(cfg))


def load_ensemble(model_dir, device):
    root = Path(model_dir)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8-sig"))
    models, norms = [], []
    for fold in range(1, 6):
        folder = root / f"fold_{fold}"
        checkpoint = folder / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}. Weights are not distributed.")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("class_names") != CLASSES:
            raise ValueError(f"Checkpoint class order must be {CLASSES}")
        model = build_model(cfg)
        model.load_state_dict(payload["model_state"], strict=True)
        models.append(model.to(device).eval())
        if cfg["USE_CLINICAL"]:
            norm = json.loads((folder / "clinical_norm.json").read_text(encoding="utf-8-sig"))
            normalize_clinical(np.zeros(15, dtype=np.float32), np.zeros(15), norm)
            norms.append(norm)
    return models, norms, cfg


@torch.inference_mode()
def predict_bag(features, models, norms, device, amp=True, clinical=None):
    x = features.to(device)
    probabilities = []
    for i, model in enumerate(models):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            if clinical is None:
                logits = model(x)
            else:
                values, mask = clinical
                values = normalize_clinical(values, mask, norms[i])
                logits = model(x, clin=torch.from_numpy(values).to(device),
                               clin_mask=torch.from_numpy(mask).to(device))
            probabilities.append(torch.softmax(logits, dim=1).squeeze(0).float().cpu())
    result = torch.stack(probabilities).mean(0).numpy()
    if not np.isfinite(result).all() or not np.isclose(result.sum(), 1, atol=1e-4):
        raise ValueError("Invalid predicted probabilities")
    return result
