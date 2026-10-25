"""Sample unused patch coordinates at the same extraction level."""
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Set, Tuple, List

import numpy as np
import pandas as pd

from .select_patches import (build_strata, allocate_quotas, FPS_FEATURES,
                             ENTROPY_Q, SOBEL_Q, LAP_Q, farthest_point_sampling)

def _select_tiles_from_dfB_aug(
    dfB: pd.DataFrame,
    total_k: int,
    seed: int,
    exclude_coords: Optional[Set[Tuple[int, int]]] = None,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    d = dfB.copy()

    if exclude_coords:
        mask_keep = ~d.apply(lambda r: (int(r["x0"]), int(r["y0"])) in exclude_coords, axis=1)
        d = d[mask_keep].reset_index(drop=True)

    if len(d) == 0:
        return d

    d = build_strata(d)
    total_k = min(int(total_k), len(d))
    quota = allocate_quotas(d, total_k)

    feat_mat = d[FPS_FEATURES].to_numpy(dtype=np.float32)
    mu = feat_mat.mean(axis=0, keepdims=True)
    sd = feat_mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    feat_z = (feat_mat - mu) / sd

    selected_rows: List[int] = []
    stratum_keys = d["stratum_key"].to_numpy()

    for s, q in quota.items():
        if q <= 0:
            continue
        rows = np.where(stratum_keys == s)[0]
        if len(rows) == 0:
            continue
        if q >= len(rows):
            selected_rows.extend(rows.tolist())
        else:
            X = feat_z[rows]

            start = int(rng.integers(0, len(rows)))
            rel = farthest_point_sampling(X, q, start=start)
            selected_rows.extend(rows[rel].tolist())

    selected_rows = list(dict.fromkeys(selected_rows))

    if len(selected_rows) < total_k:
        remain_rows = np.array([i for i in range(len(d)) if i not in set(selected_rows)], dtype=int)
        need = total_k - len(selected_rows)
        if len(remain_rows) > 0:
            Xr = feat_z[remain_rows]
            start = int(rng.integers(0, len(remain_rows)))
            rel = farthest_point_sampling(Xr, min(need, len(remain_rows)), start=start)
            selected_rows.extend(remain_rows[rel].tolist())

    if len(selected_rows) > total_k:
        Xs = feat_z[np.array(selected_rows)]
        start = int(rng.integers(0, len(selected_rows)))
        rel = farthest_point_sampling(Xs, total_k, start=start)
        selected_rows = list(np.array(selected_rows)[rel])

    df_sel = d.iloc[selected_rows].copy().reset_index(drop=True)
    return df_sel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original WSI root with class directories")
    parser.add_argument("--patches", type=Path, required=True, help="Original patch-selection output root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new augmentation output directory")
    reports = [p for p in sorted(args.patches.rglob("*_stageB_candidates.csv"))
               if p.parent.parent.name in ("PIT1", "SF1", "TPIT", "zero")]
    if not reports:
        raise FileNotFoundError("No augmentation candidates found")
    import openslide
    args.output.mkdir(parents=True)
    for report in reports:
        label, slide_name = report.parent.parent.name, report.parent.name
        files = [p for p in (args.input / label).rglob("*")
                 if p.stem == slide_name and p.suffix.lower() in (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")]
        if len(files) != 1:
            raise ValueError(f"Expected one source WSI for {label}/{slide_name}")
        original_report = report.parent / f"{slide_name}_selected_tiles.csv"
        if not original_report.is_file():
            raise FileNotFoundError(original_report)
        original = pd.read_csv(original_report)
        coordinates = set(map(tuple, original[["x0", "y0"]].to_numpy(int)))
        frame = pd.read_csv(report)
        low_texture = ((frame.entropy < frame.entropy.quantile(ENTROPY_Q)) &
                       (frame.sobel_mean < frame.sobel_mean.quantile(SOBEL_Q)) &
                       (frame.lap_var < frame.lap_var.quantile(LAP_Q)))
        frame = frame[~low_texture].reset_index(drop=True)
        # A stable digest replaces process-randomized hash() for reproducible augmentation.
        offset = int.from_bytes(hashlib.sha256(slide_name.encode()).digest()[:8], "little") % 1000003
        selected = _select_tiles_from_dfB_aug(frame, 100, args.seed + 10007 + offset, coordinates)
        if selected.empty:
            raise ValueError(f"No unused patch coordinates remain for {label}/{slide_name}")
        if len(selected) < 100:
            print(f"Warning: only {len(selected)} unused patches for {label}/{slide_name}")
        folder = args.output / label / slide_name
        tiles = folder / "tiles_selected_aug"
        tiles.mkdir(parents=True)
        paths = []
        with openslide.OpenSlide(str(files[0])) as slide:
            if not 0 <= args.level < slide.level_count:
                raise ValueError("Requested augmentation level is unavailable")
            for row in selected.itertuples():
                name = f"{slide_name}_AUG_tile_{int(row.idx):06d}.png"
                slide.read_region((int(row.x0), int(row.y0)), args.level, (256, 256)).convert("RGB").save(tiles / name)
                paths.append((Path("tiles_selected_aug") / name).as_posix())
        selected["tile_path"] = paths
        selected.to_csv(folder / f"{slide_name}_selected_tiles_aug.csv", index=False)
        print(f"Augmented {label}/{slide_name}: {len(selected)} patches")


if __name__ == "__main__":
    main()
