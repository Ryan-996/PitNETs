"""Extract Virchow2 CLS features while preserving selected-patch CSV order."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from models.common import resolve_device


class PatchDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root containing selected-patch reports")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    pattern = "*_selected_tiles_aug.csv" if args.augmented else "*_selected_tiles.csv"
    reports = sorted(args.input.rglob(pattern))
    if not reports:
        raise FileNotFoundError("No selected-patch reports found")
    if args.output.exists():
        raise FileExistsError("Choose a new feature output directory")
    device = resolve_device(args.device)
    import timm
    from timm.data import create_transform, resolve_data_config
    from timm.layers import SwiGLUPacked
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                              mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    args.output.mkdir(parents=True)
    for report in reports:
        frame = pd.read_csv(report)
        if frame.empty or not {"tile_path", "x0", "y0"}.issubset(frame):
            raise ValueError(f"Invalid selected-patch report: {report}")
        if len(frame) != 100:
            raise ValueError(f"The frozen training recipe requires 100 patches; got {len(frame)} in {report}")
        paths = [Path(p) if Path(p).is_absolute() else report.parent / p for p in frame.tile_path]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError(f"Missing patch image referenced by {report}")
        loader = DataLoader(PatchDataset(paths, transform), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False, pin_memory=True)
        batches = []
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=not args.no_amp):
                tokens = model(batch.to(device, non_blocking=True))
                features = tokens[:, 0, :]
            if features.ndim != 2 or features.shape[1] != 1280 or not torch.isfinite(features).all():
                raise ValueError("Expected finite 1280-dimensional CLS features")
            batches.append(features.half().cpu().numpy())
        relative = report.parent.relative_to(args.input)
        folder = args.output / relative.parent
        folder.mkdir(parents=True, exist_ok=True)
        name = report.parent.name + ("_aug" if args.augmented else "")
        np.save(folder / f"{name}.npy", np.concatenate(batches))
        meta = np.zeros(len(frame), dtype=[("x0", np.int32), ("y0", np.int32)])
        meta["x0"], meta["y0"] = frame.x0, frame.y0
        np.save(folder / f"{name}_meta.npy", meta)
        print(f"Extracted {name}: {len(frame)} patches")


if __name__ == "__main__":
    main()
