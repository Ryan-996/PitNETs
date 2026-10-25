"""Two-stage patch selection adapted from the final research implementation."""
from __future__ import annotations

from pathlib import Path
import math
import warnings
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image

from skimage import exposure, filters
from skimage.color import rgb2hed
from skimage.filters.rank import entropy as entropy_filter
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import distance_transform_edt

import openslide

DATA_ROOT = Path("data/wsi")
CLASS_DIRS = ["PIT1", "SF1", "TPIT", "zero", "Multi"]

OUTPUT_ROOT = Path("outputs/patches")

SKIP_IF_DONE = True

LEVEL_STAGEA_5X = 1
LEVEL_20X = 1

TILE_SIZE = (256, 256)

N_SAMPLED_PATCHES = 100
TOP_STAGEB_CANDIDATES = 8000

MEAN_GRAY_Q = 0.90
STD_GRAY_Q = 0.10

ENTROPY_Q = 0.20
SOBEL_Q = 0.20
LAP_Q = 0.20

STRATA_FEATURES = ["H_mean", "E_mean", "pink_purple_ratio", "nuclei_density_log"]
N_BINS_PER_FEAT = 3

RARE_STRATUM_MIN_COUNT = 12
RARE_STRATUM_BONUS = 1

FPS_FEATURES = [
    "H_mean", "E_mean", "pink_purple_ratio",
    "entropy", "sobel_mean", "lap_var",
    "glcm_contrast", "glcm_energy", "glcm_homogeneity",
    "nuclei_density_log"
]

CORE_EDGE_BINS = (0.33, 0.66)

STAGEA_WORKERS = max(1, min(16, (os.cpu_count() or 8) // 2))
STAGEA_CHUNK = 128

SEED = 42

warnings.filterwarnings(
    "ignore",
    message=(
        "Input image must be RGB. NOTE: the image will be converted to RGB before "
        "HED conversion."
    ),
    category=UserWarning,
)

def _winsorize(x: np.ndarray, p_low=0.01, p_high=0.99) -> np.ndarray:
    lo = np.quantile(x, p_low)
    hi = np.quantile(x, p_high)
    return np.clip(x, lo, hi)

def core_edge_bin_from_dist(dist01: float) -> str:
    a, b = CORE_EDGE_BINS
    if dist01 < a:
        return "edge"
    elif dist01 < b:
        return "mid"
    else:
        return "core"

def build_thumbnail_tissue_distance_map(
    osr: openslide.OpenSlide,
    thumb_level: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    used_level = (osr.level_count - 1) if thumb_level is None else int(thumb_level)
    w, h = osr.level_dimensions[used_level]
    down = float(osr.level_downsamples[used_level])

    thumb = osr.read_region((0, 0), used_level, (w, h)).convert("RGB")
    thumb_np = np.array(thumb)

    gray = exposure.rescale_intensity(
        (0.299 * thumb_np[:, :, 0] + 0.587 * thumb_np[:, :, 1] + 0.114 * thumb_np[:, :, 2]),
        out_range=(0, 1),
    )
    inv = 1.0 - gray
    thr = filters.threshold_otsu(inv)
    tissue = inv > thr

    tissue = remove_small_holes(tissue, area_threshold=256)
    tissue = remove_small_objects(tissue, min_size=256)

    dist = distance_transform_edt(tissue)
    dist_norm = np.zeros_like(dist, dtype=np.float32)
    if dist.max() > 0:
        dist_norm[tissue] = (dist[tissue] / dist[tissue].max()).astype(np.float32)

    return tissue.astype(bool), dist_norm, down, used_level

@dataclass(frozen=True)
class StageATask:
    idx: int
    x0: int
    y0: int
    core_edge: str
    dist01: float

_WSI_PATH_GLOBAL: str | None = None
_LEVEL_STAGEA_GLOBAL: int | None = None
_TILE_SIZE_GLOBAL: Tuple[int, int] | None = None

def _stagea_worker_init(wsi_path: str, level_stagea: int, tile_size: Tuple[int, int]) -> None:
    global _WSI_PATH_GLOBAL, _LEVEL_STAGEA_GLOBAL, _TILE_SIZE_GLOBAL
    _WSI_PATH_GLOBAL = wsi_path
    _LEVEL_STAGEA_GLOBAL = level_stagea
    _TILE_SIZE_GLOBAL = tile_size

def _compute_stagea_cheap_one(task: StageATask) -> Dict[str, Any]:
    assert _WSI_PATH_GLOBAL is not None
    assert _LEVEL_STAGEA_GLOBAL is not None
    assert _TILE_SIZE_GLOBAL is not None

    osr = openslide.OpenSlide(_WSI_PATH_GLOBAL)
    try:
        img_l = osr.read_region((task.x0, task.y0), _LEVEL_STAGEA_GLOBAL, _TILE_SIZE_GLOBAL).convert("L")
        gray8 = np.array(img_l, dtype=np.uint8)

        mean_gray = float(gray8.mean())
        std_gray = float(gray8.std())

        gray_f = gray8.astype(np.float32) / 255.0
        sob = filters.sobel(gray_f)
        sobel_mean = float(sob.mean())

        lap = filters.laplace(gray8.astype(np.float32))
        lap_var = float(lap.var())

        return {
            "idx": task.idx,
            "x0": task.x0,
            "y0": task.y0,
            "core_edge": task.core_edge,
            "dist01": task.dist01,
            "mean_gray": mean_gray,
            "std_gray": std_gray,
            "lap_var_A": lap_var,
            "sobel_mean_A": sobel_mean,
        }
    finally:
        osr.close()

def compute_stageB_heavy_features_from_rgb(arr_rgb: np.ndarray) -> dict:
    hed = rgb2hed(arr_rgb)
    H = hed[:, :, 0]
    E = hed[:, :, 1]
    Hn = exposure.rescale_intensity(H, out_range=(0, 1))
    En = exposure.rescale_intensity(E, out_range=(0, 1))

    H_mean = float(Hn.mean())
    E_mean = float(En.mean())
    pink_purple_ratio = float((E_mean + 1e-6) / (H_mean + 1e-6))

    gray8 = np.array(Image.fromarray(arr_rgb).convert("L"), dtype=np.uint8)

    gray128 = gray8[::2, ::2]
    ent_map = entropy_filter(gray128, disk(3))
    entropy_mean = float(ent_map.mean())

    lap = filters.laplace(gray8.astype(np.float32))
    lap_var = float(lap.var())
    sob = filters.sobel(gray8.astype(np.float32) / 255.0)
    sobel_mean = float(sob.mean())

    try:
        thr_h = filters.threshold_otsu(Hn)
    except Exception:
        thr_h = float(np.quantile(Hn, 0.75))
    nuclei = Hn > thr_h
    nuclei = remove_small_objects(nuclei, min_size=12)
    nuclei_density = float(nuclei.mean())

    gray64 = gray8[::4, ::4]
    gray32 = (gray64 // 8).astype(np.uint8)
    glcm = graycomatrix(gray32, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    glcm_contrast = float(graycoprops(glcm, "contrast")[0, 0])
    glcm_energy = float(graycoprops(glcm, "energy")[0, 0])
    glcm_homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])

    return {
        "H_mean": H_mean,
        "E_mean": E_mean,
        "pink_purple_ratio": pink_purple_ratio,
        "entropy": entropy_mean,
        "sobel_mean": sobel_mean,
        "lap_var": lap_var,
        "nuclei_density": nuclei_density,
        "glcm_contrast": glcm_contrast,
        "glcm_energy": glcm_energy,
        "glcm_homogeneity": glcm_homogeneity,
    }

def build_strata(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    nd = np.log1p(d["nuclei_density"].to_numpy())
    nd = _winsorize(nd, 0.01, 0.99)
    d["nuclei_density_log"] = nd

    for feat in STRATA_FEATURES:
        try:
            d[f"{feat}_bin"] = pd.qcut(d[feat], q=N_BINS_PER_FEAT, labels=False, duplicates="drop")
        except Exception:
            d[f"{feat}_bin"] = pd.cut(d[feat], bins=N_BINS_PER_FEAT, labels=False, include_lowest=True)
        d[f"{feat}_bin"] = d[f"{feat}_bin"].astype("Int64")

    bins_cols = [f"{f}_bin" for f in STRATA_FEATURES]
    d["stratum_key"] = d["core_edge"].astype(str) + "_" + d[bins_cols].astype(str).agg("-".join, axis=1)
    return d

def allocate_quotas(df: pd.DataFrame, total_k: int) -> dict[str, int]:
    counts = df["stratum_key"].value_counts()
    strata = counts.index.tolist()
    n_total = int(counts.sum())

    raw = {s: (counts[s] / n_total) * total_k for s in strata}
    quota = {s: int(math.floor(raw[s])) for s in strata}

    used = sum(quota.values())
    remain = total_k - used
    frac_sorted = sorted(strata, key=lambda s: raw[s] - quota[s], reverse=True)
    for s in frac_sorted:
        if remain <= 0:
            break
        quota[s] += 1
        remain -= 1

    rare = [s for s in strata if counts[s] < RARE_STRATUM_MIN_COUNT]
    for s in rare:
        if sum(quota.values()) >= total_k:
            break
        quota[s] += RARE_STRATUM_BONUS

    while sum(quota.values()) > total_k:
        donors = [s for s in strata if s not in rare and quota[s] > 0]
        if not donors:
            donors = [s for s in strata if quota[s] > 0]
        donor = max(donors, key=lambda s: quota[s])
        quota[donor] -= 1

    for s in strata:
        quota[s] = min(quota[s], int(counts[s]))

    deficit = total_k - sum(quota.values())
    if deficit > 0:
        donors = sorted(strata, key=lambda s: counts[s], reverse=True)
        for s in donors:
            if deficit <= 0:
                break
            cap = int(counts[s]) - quota[s]
            if cap <= 0:
                continue
            add = min(cap, deficit)
            quota[s] += add
            deficit -= add

    return quota

def farthest_point_sampling(X: np.ndarray, k: int, start: int = 0) -> np.ndarray:
    n = X.shape[0]
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n)
    start = int(start) % n
    selected = [start]
    dist_min = np.full(n, np.inf, dtype=np.float64)
    dist_min[start] = -np.inf
    for _ in range(1, k):
        last = selected[-1]
        d = np.sum((X - X[last]) ** 2, axis=1)
        dist_min = np.minimum(dist_min, d)
        nxt = int(np.argmax(dist_min))
        selected.append(nxt)
        dist_min[nxt] = -np.inf
    return np.array(selected, dtype=int)

def process_one_wsi(wsi_path: Path, label_name: str) -> None:
    slide_name = wsi_path.stem

    sample_out = OUTPUT_ROOT / label_name / slide_name
    out_tile_dir = sample_out / "tiles_selected"
    out_tile_dir.mkdir(parents=True, exist_ok=True)

    cheap_csv = sample_out / f"{slide_name}_stageA_cheap.csv"
    allfeat_csv = sample_out / f"{slide_name}_stageB_candidates.csv"
    report_csv = sample_out / f"{slide_name}_selected_tiles.csv"

    if SKIP_IF_DONE and report_csv.exists():
        print(f"[SKIP] {label_name}/{slide_name} already done: {report_csv}")
        return

    osr0 = openslide.OpenSlide(str(wsi_path))

    tissue_thumb, dist_map, down_thumb, used_thumb_level = build_thumbnail_tissue_distance_map(osr0, thumb_level=None)

    down_stageA = float(osr0.level_downsamples[LEVEL_STAGEA_5X])
    down_20x = float(osr0.level_downsamples[LEVEL_20X])

    stageA_to_thumb = down_stageA / down_thumb

    wA, hA = osr0.level_dimensions[LEVEL_STAGEA_5X]
    step_x, step_y = TILE_SIZE

    print(f"\n[INFO] === {label_name}/{slide_name} ===")
    print(f"[INFO] WSI: {wsi_path}")
    print(f"[INFO] StageA level={LEVEL_STAGEA_5X} dims={wA}x{hA}, down={down_stageA}")
    print(f"[INFO] StageB level={LEVEL_20X} down={down_20x}")
    print(f"[INFO] thumb level used: L{used_thumb_level}, down={down_thumb}")
    print(f"[INFO] workers={STAGEA_WORKERS}, chunk={STAGEA_CHUNK}")
    print(f"[INFO] topK={TOP_STAGEB_CANDIDATES}, final={N_SAMPLED_PATCHES}")

    tasks: List[StageATask] = []
    idx = 0

    for yA in range(0, hA - TILE_SIZE[1] + 1, step_y):
        cyA = yA + TILE_SIZE[1] / 2.0
        ty = int(cyA * stageA_to_thumb)
        if ty < 0 or ty >= tissue_thumb.shape[0]:
            continue

        for xA in range(0, wA - TILE_SIZE[0] + 1, step_x):
            cxA = xA + TILE_SIZE[0] / 2.0
            tx = int(cxA * stageA_to_thumb)
            if tx < 0 or tx >= tissue_thumb.shape[1]:
                continue

            if not tissue_thumb[ty, tx]:
                continue

            dist01 = float(dist_map[ty, tx])
            core_edge = core_edge_bin_from_dist(dist01)

            x0 = int(xA * down_stageA)
            y0 = int(yA * down_stageA)
            tasks.append(StageATask(idx=idx, x0=x0, y0=y0, core_edge=core_edge, dist01=dist01))
            idx += 1

    osr0.close()

    print(f"[INFO] StageA tasks(after tissue-skip): {len(tasks)}")
    if len(tasks) == 0:
        print(f"[WARN] {label_name}/{slide_name}: no tasks, skip.")
        return

    print("[INFO] StageA(mp): extracting cheap features...")
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=STAGEA_WORKERS,
        initializer=_stagea_worker_init,
        initargs=(str(wsi_path), LEVEL_STAGEA_5X, TILE_SIZE),
        maxtasksperchild=256,
    ) as pool:
        results_iter = pool.imap_unordered(_compute_stagea_cheap_one, tasks, chunksize=STAGEA_CHUNK)
        stageA_records = list(tqdm(results_iter, total=len(tasks), desc=f"{slide_name} StageA", unit="tile"))

    dfA = pd.DataFrame(stageA_records).sort_values("idx").reset_index(drop=True)
    dfA["label"] = label_name
    dfA["slide"] = slide_name
    dfA.to_csv(cheap_csv, index=False, encoding="utf-8-sig")

    mean_thr = dfA["mean_gray"].quantile(MEAN_GRAY_Q)
    std_thr = dfA["std_gray"].quantile(STD_GRAY_Q)
    dfA_f = dfA[(dfA["mean_gray"] < mean_thr) & (dfA["std_gray"] > std_thr)].copy()
    if len(dfA_f) == 0:
        dfA_f = dfA.copy()

    lapA = _winsorize(dfA_f["lap_var_A"].to_numpy(), 0.01, 0.99)
    sobA = _winsorize(dfA_f["sobel_mean_A"].to_numpy(), 0.01, 0.99)
    lapZ = (lapA - lapA.mean()) / (lapA.std() + 1e-8)
    sobZ = (sobA - sobA.mean()) / (sobA.std() + 1e-8)
    dfA_f["scoreA"] = lapZ + sobZ

    topK = min(TOP_STAGEB_CANDIDATES, len(dfA_f))
    dfA_top = dfA_f.nlargest(topK, "scoreA").reset_index(drop=True)
    print(f"[INFO] StageA kept for StageB: {len(dfA_top)} / {len(dfA)}")

    osr = openslide.OpenSlide(str(wsi_path))
    print("[INFO] StageB: computing heavy features...")
    stageB_records: List[Dict[str, Any]] = []

    for r in tqdm(dfA_top.itertuples(index=False), total=len(dfA_top), desc=f"{slide_name} StageB", unit="tile"):
        x0, y0 = int(r.x0), int(r.y0)
        img = osr.read_region((x0, y0), LEVEL_20X, TILE_SIZE).convert("RGB")
        arr = np.array(img)
        heavy = compute_stageB_heavy_features_from_rgb(arr)

        stageB_records.append({
            "label": label_name,
            "slide": slide_name,
            "idx": int(r.idx),
            "x0": x0,
            "y0": y0,
            "core_edge": r.core_edge,
            "dist01": float(r.dist01),
            "mean_gray": float(r.mean_gray),
            "std_gray": float(r.std_gray),
            "lap_var_A": float(r.lap_var_A),
            "sobel_mean_A": float(r.sobel_mean_A),
            **heavy,
        })

    dfB = pd.DataFrame(stageB_records)
    dfB.to_csv(allfeat_csv, index=False, encoding="utf-8-sig")

    if len(dfB) == 0:
        print(f"[WARN] {label_name}/{slide_name}: no stageB records.")
        osr.close()
        return

    ent_thr = dfB["entropy"].quantile(ENTROPY_Q)
    sob_thr = dfB["sobel_mean"].quantile(SOBEL_Q)
    lap_thr = dfB["lap_var"].quantile(LAP_Q)
    low_texture = (dfB["entropy"] < ent_thr) & (dfB["sobel_mean"] < sob_thr) & (dfB["lap_var"] < lap_thr)
    dfB = dfB[~low_texture].reset_index(drop=True)

    if len(dfB) == 0:
        print(f"[WARN] {label_name}/{slide_name}: all filtered out at StageB.")
        osr.close()
        return

    dfB = build_strata(dfB)
    total_k = min(N_SAMPLED_PATCHES, len(dfB))
    quota = allocate_quotas(dfB, total_k)

    feat_mat = dfB[FPS_FEATURES].to_numpy(dtype=np.float32)
    mu = feat_mat.mean(axis=0, keepdims=True)
    sd = feat_mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    feat_z = (feat_mat - mu) / sd

    selected_rows: List[int] = []
    for s, q in quota.items():
        if q <= 0:
            continue
        rows = np.where(dfB["stratum_key"].to_numpy() == s)[0]
        if len(rows) == 0:
            continue
        if q >= len(rows):
            selected_rows.extend(rows.tolist())
        else:
            X = feat_z[rows]
            rel = farthest_point_sampling(X, q)
            selected_rows.extend(rows[rel].tolist())

    selected_rows = list(dict.fromkeys(selected_rows))
    if len(selected_rows) < total_k:
        remain_rows = np.array([i for i in range(len(dfB)) if i not in set(selected_rows)], dtype=int)
        need = total_k - len(selected_rows)
        if len(remain_rows) > 0:
            Xr = feat_z[remain_rows]
            rel = farthest_point_sampling(Xr, min(need, len(remain_rows)))
            selected_rows.extend(remain_rows[rel].tolist())

    if len(selected_rows) > total_k:
        Xs = feat_z[np.array(selected_rows)]
        rel = farthest_point_sampling(Xs, total_k)
        selected_rows = list(np.array(selected_rows)[rel])

    df_sel = dfB.iloc[selected_rows].copy().reset_index(drop=True)

    print("[INFO] Saving selected tiles...")
    out_paths: List[str] = []
    for row in tqdm(df_sel.itertuples(index=False), total=len(df_sel), desc=f"{slide_name} Save", unit="tile"):
        x0, y0 = int(row.x0), int(row.y0)
        img = osr.read_region((x0, y0), LEVEL_20X, TILE_SIZE).convert("RGB")

        fname = (
            f"{slide_name}_tile_{int(row.idx):06d}_L{LEVEL_20X}_{row.core_edge}"
            f"_H{row.H_mean:.3f}_E{row.E_mean:.3f}_nd{row.nuclei_density:.4f}.png"
        )
        fpath = out_tile_dir / fname
        img.save(fpath)
        out_paths.append(fpath.relative_to(sample_out).as_posix())

    df_sel["tile_path"] = out_paths
    df_sel.to_csv(report_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {label_name}/{slide_name} done -> {report_csv}")

    osr.close()

def iter_wsi_files() -> List[tuple[str, Path]]:
    items: List[tuple[str, Path]] = []
    for cls in CLASS_DIRS:
        cls_dir = DATA_ROOT / cls
        if not cls_dir.exists():
            print(f"[WARN] class dir not found: {cls_dir}")
            continue

        for ext in ("*.svs", "*.tif", "*.tiff", "*.ndpi", "*.mrxs"):
            for fp in cls_dir.rglob(ext):
                items.append((cls, fp))

    items.sort(key=lambda x: (x[0], x[1].name))
    return items

def main() -> None:
    import argparse
    global DATA_ROOT, OUTPUT_ROOT, LEVEL_STAGEA_5X, LEVEL_20X, STAGEA_WORKERS
    parser = argparse.ArgumentParser(description="Two-stage pathology-informed patch selection")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-a-level", type=int, default=1)
    parser.add_argument("--stage-b-level", type=int, default=1)
    parser.add_argument("--workers", type=int, default=STAGEA_WORKERS)
    args = parser.parse_args()
    if not args.input.is_dir() or args.workers < 1:
        raise ValueError("Input must be a directory and workers must be positive")
    if args.input.resolve() == args.output.resolve() or args.input.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be outside the input directory")
    DATA_ROOT, OUTPUT_ROOT = args.input, args.output
    LEVEL_STAGEA_5X, LEVEL_20X = args.stage_a_level, args.stage_b_level
    STAGEA_WORKERS = args.workers
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_items = iter_wsi_files()
    print(f"[INFO] total slides found: {len(all_items)}")
    if len(all_items) == 0:
        raise ValueError("No WSI found in the five class directories")
    failed: List[tuple[str, Path, str]] = []

    for cls, wsi_path in all_items:
        try:
            process_one_wsi(wsi_path=wsi_path, label_name=cls)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {cls}/{wsi_path.name} -> {msg}")
            failed.append((cls, wsi_path, msg))
            continue

    if failed:
        fail_csv = OUTPUT_ROOT / "failed_list.csv"
        pd.DataFrame(
            [{"label": c, "wsi": str(p), "error": m} for c, p, m in failed]
        ).to_csv(fail_csv, index=False, encoding="utf-8-sig")
        raise RuntimeError(f"Failed {len(failed)} slides; details: {fail_csv}")
    else:
        print("[INFO] all slides processed successfully.")

if __name__ == "__main__":
    main()
