# -*- coding: utf-8 -*-
"""
PitNETs MIL 模型（HE图像特征 + 临床信息，图像条件化临床 gating 版）

满足用户修改要求：
1) 临床仅两个连续变量，其余严格二分类(0/1)；
   ClinicalTable 初始化只做解析：standardize_continuous=False 时不fit均值方差、不做归一化；
   归一化由训练脚本对训练集完成，并导出参数（本脚本不包含fit逻辑）。
2) 图像条件化 gating：使用 z_img 条件化生成每个临床特征 gate（15维），实现图像-临床协同。

使用方式：
- 从 npy 读出 HE patch 特征 x
- ClinicalTable + _extract_pid_from_name 匹配 clin(15) 与 mask(15)
- forward(x, clin, mask) 输出 logits(5)
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List
import re

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# =====================================================
# 0. PID 解析：严格使用用户给定逻辑
# =====================================================
def _extract_pid_from_name(name: str, known_ids: set, pid_len: int = 7) -> Optional[str]:
    s = str(name)
    runs = re.findall(r"\d+", s)
    runs = [r for r in runs if len(r) >= pid_len]
    if not runs:
        return None
    for r in runs:
        for i in range(0, len(r) - pid_len + 1):
            cand = r[i:i + pid_len]
            if cand in known_ids:
                return cand
    return None


def _collect_known_ids_from_pid_col(pid_series: pd.Series, pid_len: int = 7) -> set:
    """
    从 clinical.csv 第1列（PID列）收集 known_ids。
    - 支持 PID 单元格里含有非数字字符（如 'P1234567'）
    - 只收集长度==pid_len 的数字串
    """
    known = set()
    for v in pid_series.astype(str).tolist():
        runs = re.findall(r"\d+", str(v))
        for r in runs:
            if len(r) == pid_len:
                known.add(r)
            elif len(r) > pid_len:
                for i in range(0, len(r) - pid_len + 1):
                    known.add(r[i:i + pid_len])
    return known


# =====================================================
# 1. Clinical 表读取、编码（不做fit/标准化）
# =====================================================
class ClinicalTable:
    """
    读取 clinical.csv（16列：第1列PID用于match，其余15列为特征）
    - known_ids: 用于从 npy 文件名中提取 PID 的集合
    - get_by_pid(pid) -> (clin(15,), mask(15,))
    - standardize_continuous=False 时：初始化只解析，不fit、不标准化
    """

    def __init__(
        self,
        csv_path: str,
        pid_len: int = 7,
        # 除PID外共15个特征：columns[1:16]
        # 默认：年龄为第3列 => 在15维特征里 index=1；肿瘤直径为第16列 => index=14
        continuous_feature_indices: Tuple[int, int] = (1, 14),
        standardize_continuous: bool = False,  # <<< 按需求：默认关闭
    ) -> None:
        self.csv_path = csv_path
        self.pid_len = int(pid_len)
        self.continuous_feature_indices = tuple(int(i) for i in continuous_feature_indices)
        self.standardize_continuous = bool(standardize_continuous)

        self.df = pd.read_csv(csv_path)
        if self.df.shape[1] != 16:
            raise ValueError(f"clinical.csv 期望 16 列，但读取到 {self.df.shape[1]} 列：{csv_path}")

        self.pid_col_name = self.df.columns[0]
        self.feature_col_names = list(self.df.columns[1:16])  # 15列特征

        self.known_ids = _collect_known_ids_from_pid_col(self.df[self.pid_col_name], pid_len=self.pid_len)

        # 建立 pid -> row 索引映射（提取7位数字作为PID key）
        pid_keys: List[Optional[str]] = []
        for raw in self.df[self.pid_col_name].astype(str).tolist():
            runs = re.findall(r"\d+", str(raw))
            key = None
            for r in runs:
                if len(r) == self.pid_len:
                    key = r
                    break
            if key is None and runs:
                r0 = runs[0]
                if len(r0) > self.pid_len:
                    key = r0[: self.pid_len]
            pid_keys.append(key)

        self._pid_to_index: Dict[str, int] = {}
        for idx, k in enumerate(pid_keys):
            if k is not None and k not in self._pid_to_index:
                self._pid_to_index[k] = idx

        # 注意：按需求不在这里fit均值方差；归一化由训练脚本处理

    @staticmethod
    def _to_binary_strict_01(v: Any) -> Tuple[float, float]:
        """
        严格二分类：只接受 0 或 1
        - 返回 (value, mask)；mask=1 有效，mask=0 缺失或非法
        """
        if pd.isna(v):
            return 0.0, 0.0

        # bool 也允许
        if isinstance(v, (bool, np.bool_)):
            return (1.0 if bool(v) else 0.0), 1.0

        s = str(v).strip()
        if s == "":
            return 0.0, 0.0

        try:
            x = float(s)
        except Exception:
            return 0.0, 0.0

        if x == 0.0:
            return 0.0, 1.0
        if x == 1.0:
            return 1.0, 1.0
        return 0.0, 0.0

    @staticmethod
    def _to_continuous(v: Any) -> Tuple[float, float]:
        """
        连续变量：返回 (value, mask)
        归一化不在此处做（由训练脚本完成后把值传入/或预处理后保存）
        """
        if pd.isna(v):
            return 0.0, 0.0
        try:
            x = float(v)
        except Exception:
            s = str(v).strip()
            if s == "":
                return 0.0, 0.0
            try:
                x = float(s)
            except Exception:
                return 0.0, 0.0
        return float(x), 1.0

    def encode_row(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        将一行临床信息编码为 (clin(15,), mask(15,))
        - 连续变量：原值输出（不标准化）
        - 其余：严格0/1
        """
        clin = np.zeros((15,), dtype=np.float32)
        mask = np.zeros((15,), dtype=np.float32)

        for i, col in enumerate(self.feature_col_names):
            if i in self.continuous_feature_indices:
                x, m = self._to_continuous(row[col])
                clin[i] = float(x)
                mask[i] = float(m)
            else:
                x, m = self._to_binary_strict_01(row[col])
                clin[i] = float(x)
                mask[i] = float(m)

        return clin, mask

    def get_by_pid(self, pid: str) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._pid_to_index.get(str(pid), None)
        if idx is None:
            return np.zeros((15,), dtype=np.float32), np.zeros((15,), dtype=np.float32)
        row = self.df.iloc[idx]
        return self.encode_row(row)

    def get_by_npy_name(self, npy_name_or_path: str) -> Tuple[Optional[str], np.ndarray, np.ndarray]:
        pid = _extract_pid_from_name(npy_name_or_path, self.known_ids, pid_len=self.pid_len)
        if pid is None:
            return None, np.zeros((15,), dtype=np.float32), np.zeros((15,), dtype=np.float32)
        clin, mask = self.get_by_pid(pid)
        return pid, clin, mask


def default_part_slices_for_15_features() -> Dict[str, slice]:
    """
    15维临床特征默认分组（按你描述的列结构）：
    - 2-3列：病人信息 -> 2维 => feature[0:2]
    - 4-9列：临床症状 -> 6维 => feature[2:8]
    - 10-15列：血清学 -> 6维 => feature[8:14]
    - 16列：肿瘤直径 -> 1维 => feature[14:15]
    """
    return {
        "patient_info": slice(0, 2),
        "symptoms": slice(2, 8),
        "serology": slice(8, 14),
        "tumor_diameter": slice(14, 15),
    }


# =====================================================
# 2. Gated Attention（patch-level）
# =====================================================
class GatedAttention(nn.Module):
    def __init__(self, d_in: int, d_attn: int = 128, dropout: float = 0.25):
        super().__init__()
        self.attn_v = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_u = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_w = nn.Linear(d_attn, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N,D) or (B,N,D)
        if h.dim() == 2:
            v = self.attn_v(h)
            u = self.attn_u(h)
            a = self.attn_w(v * u).squeeze(-1)  # (N,)
            return torch.softmax(a, dim=0)
        elif h.dim() == 3:
            v = self.attn_v(h)
            u = self.attn_u(h)
            a = self.attn_w(v * u).squeeze(-1)  # (B,N)
            return torch.softmax(a, dim=1)
        else:
            raise ValueError("Invalid input dim for GatedAttention")


# =====================================================
# 3. 图像条件化临床 gating（Image-conditioned Clinical Gating）
# =====================================================
class ImageConditionedClinicalGating(nn.Module):
    """
    使用 z_img 条件化，输出每个临床特征 gate（B, C），再与 clin/mask 组合：
    - gate = sigmoid(MLP([z_img, clin, mask]))
    - 对缺失特征：gate强制为0（避免缺失影响）
    输出：
    - clin_gated = clin * gate
    - gate (便于解释/可视化)
    """
    def __init__(
        self,
        img_dim: int,
        clin_dim: int = 15,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.clin_dim = int(clin_dim)
        in_dim = int(img_dim) + int(clin_dim) * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, clin_dim),
        )

    def forward(
        self,
        z_img: torch.Tensor,       # (B, img_dim)
        clin: torch.Tensor,        # (B, C)
        clin_mask: torch.Tensor,   # (B, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_img.dim() != 2:
            raise ValueError("z_img must be (B, img_dim)")
        if clin.dim() != 2:
            raise ValueError("clin must be (B, C)")
        if clin_mask.dim() != 2:
            raise ValueError("clin_mask must be (B, C)")

        x = torch.cat([z_img, clin, clin_mask], dim=1)
        gate_logits = self.net(x)
        gate = torch.sigmoid(gate_logits)

        # 缺失特征 gate=0
        gate = gate * (clin_mask >= 0.5).to(gate.dtype)

        clin_gated = clin * gate
        return clin_gated, gate


# =====================================================
# 4. Part-level 编码器
# =====================================================
class PartClinicalEncoderLN(nn.Module):
    def __init__(self, part_dim: int, hid: int = 64, out_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(part_dim * 2, hid)
        self.ln1 = nn.LayerNorm(hid)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.act2 = nn.GELU()

    def forward(self, clin_part, mask_part):
        x = torch.cat([clin_part, mask_part], dim=-1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        return x


# =====================================================
# 5. Structured Clinical Encoder
# =====================================================
class StructuredClinicalEncoderAdvanced(nn.Module):
    def __init__(
        self,
        part_dims: Dict[str, int],
        part_hid: int = 64,
        part_out: int = 64,
        dropout: float = 0.25,
        n_heads: int = 2,
        attn_ff: int = 128,
        clin_out: int = 128,
        num_layers: int = 1,   # <<< NEW
    ):
        super().__init__()
        self.part_names = list(part_dims.keys())
        self.part_encoders = nn.ModuleDict({
            n: PartClinicalEncoderLN(part_dims[n], part_hid, part_out, dropout)
            for n in self.part_names
        })

        self.part_gate_logits = nn.Parameter(torch.zeros(len(self.part_names)))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=part_out,
            nhead=n_heads,
            dim_feedforward=attn_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.cross_part = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))


        self.pool_norm = nn.LayerNorm(part_out)
        self.fuse = nn.Sequential(
            nn.Linear(part_out, clin_out),
            nn.LayerNorm(clin_out),
            nn.GELU(),
        )

    def forward(self, clin, mask, part_slices):
        single = False
        if clin.dim() == 1:
            clin = clin.unsqueeze(0)
            mask = mask.unsqueeze(0)
            single = True

        tokens = []
        for n in self.part_names:
            sl = part_slices[n]
            z = self.part_encoders[n](clin[:, sl], mask[:, sl])
            tokens.append(z)

        tokens = torch.stack(tokens, dim=1)  # (B,P,D)
        part_gates = torch.sigmoid(self.part_gate_logits)
        tokens = tokens * part_gates.view(1, -1, 1)

        tokens = self.cross_part(tokens)
        pooled = self.pool_norm(tokens.mean(dim=1))
        z_clin = self.fuse(pooled)

        if single:
            return z_clin.squeeze(0), part_gates
        return z_clin, part_gates


# =====================================================
# 6. Helper: classifier head builder
# =====================================================
def build_classifier_head(
    in_dim: int,
    n_classes: int,
    cls_hidden_dim: int = 256,
    cls_layers: int = 1,
    dropout: float = 0.25,
) -> nn.Module:
    cls_layers = int(cls_layers)
    if cls_layers <= 1:
        return nn.Linear(in_dim, n_classes)

    layers = []
    dim = in_dim
    for _ in range(cls_layers - 1):
        layers += [
            nn.Linear(dim, cls_hidden_dim),
            nn.LayerNorm(cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        dim = cls_hidden_dim

    layers.append(nn.Linear(dim, n_classes))
    return nn.Sequential(*layers)


# =====================================================
# 7. 主模型 PitNETsMIL（图像条件化临床 gating）
# =====================================================
class PitNETsMIL(nn.Module):
    def __init__(
        self,
        n_classes: int = 5,
        in_dim: int = 1280,
        embed_dim: int = 256,
        dropout: float = 0.25,

        use_clinical: bool = True,

        part_slices: Optional[Dict[str, slice]] = None,
        part_dims: Optional[Dict[str, int]] = None,

        clin_out: int = 128,

        # --------- 可搜索超参 ----------
        attn_dim: int = 128,
        attn_dropout: Optional[float] = None,
        cls_hidden_dim: int = 256,
        cls_layers: int = 1,
        instance_dropout: float = 0.0,

        # gating超参
        clin_dim: int = 15,
        gating_hidden: int = 128,
        gating_dropout: float = 0.1,

        # ===== NEW: train-time clinical mask augmentation =====
        clinical_feature_mask_drop: float = 0.0,  # feature-level dropout prob
        clinical_part_mask_drop: float = 0.0,     # part-level dropout prob
        # ===== NEW: clinical encoder hyperparams =====
        clin_part_hid: int = 64,
        clin_part_out: int = 64,
        clin_enc_dropout: float = 0.25,
        clin_n_heads: int = 2,
        clin_attn_ff: int = 128,
        clin_num_layers: int = 1,

    ):
        super().__init__()
        self.use_clinical = bool(use_clinical)

        # ---- Image MIL backbone ----
        self.embed_img = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        if attn_dropout is None:
            attn_dropout = dropout
        self.attn = GatedAttention(embed_dim, d_attn=int(attn_dim), dropout=float(attn_dropout))

        self.instance_dropout = float(instance_dropout)
        # ---- NEW: Clinical mask augmentation params ----
        self.clinical_feature_mask_drop = float(clinical_feature_mask_drop)
        self.clinical_part_mask_drop = float(clinical_part_mask_drop)
        # ---- Clinical branch ----
        if self.use_clinical:
            if part_slices is None:
                part_slices = default_part_slices_for_15_features()
            self.part_slices = part_slices

            if part_dims is None:
                part_dims = {k: (v.stop - v.start) for k, v in part_slices.items()}

            # 图像条件化 gating
            self.clin_gating = ImageConditionedClinicalGating(
                img_dim=embed_dim,
                clin_dim=int(clin_dim),
                hidden=int(gating_hidden),
                dropout=float(gating_dropout),
            )

            self.clin_enc = StructuredClinicalEncoderAdvanced(
                part_dims=part_dims,
                part_hid=int(clin_part_hid),
                part_out=int(clin_part_out),
                dropout=float(clin_enc_dropout),
                n_heads=int(clin_n_heads),
                attn_ff=int(clin_attn_ff),
                clin_out=int(clin_out),
                num_layers=int(clin_num_layers),
            )

            head_in = embed_dim + clin_out
            self.classifier = build_classifier_head(
                in_dim=head_in,
                n_classes=n_classes,
                cls_hidden_dim=int(cls_hidden_dim),
                cls_layers=int(cls_layers),
                dropout=float(dropout),
            )
        else:
            head_in = embed_dim
            self.classifier = build_classifier_head(
                in_dim=head_in,
                n_classes=n_classes,
                cls_hidden_dim=int(cls_hidden_dim),
                cls_layers=int(cls_layers),
                dropout=float(dropout),
            )

    def _apply_instance_dropout(self, h: torch.Tensor) -> torch.Tensor:
        p = self.instance_dropout
        if (not self.training) or p <= 0.0:
            return h

        if h.dim() == 2:
            N = h.size(0)
            keep = (torch.rand(N, device=h.device) > p)
            if keep.sum() == 0:
                keep[torch.randint(0, N, (1,), device=h.device)] = True
            return h * keep.unsqueeze(-1).to(h.dtype)

        if h.dim() == 3:
            B, N, _ = h.shape
            keep = (torch.rand(B, N, device=h.device) > p)
            none = (keep.sum(dim=1) == 0)
            if none.any():
                idx = torch.randint(0, N, (int(none.sum().item()),), device=h.device)
                keep[none, idx] = True
            return h * keep.unsqueeze(-1).to(h.dtype)

        return h
    def _augment_clinical_masks(
        self,
        clin: torch.Tensor,       # (B,C)
        clin_mask: torch.Tensor,  # (B,C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练时临床mask增强：
        1) feature-level mask dropout: 随机把部分特征mask置0
        2) part-level mask dropout: 随机把某些part整组mask置0

        返回：
        - clin_aug: (B,C)（被drop的特征置0）
        - mask_aug: (B,C)
        """
        # 推理/验证阶段不增强
        if not self.training:
            return clin, clin_mask

        mask_aug = clin_mask

        # ---- feature-level dropout ----
        p_feat = self.clinical_feature_mask_drop
        if p_feat > 0.0:
            B, C = clin.shape
            keep = (torch.rand(B, C, device=clin.device) > p_feat).to(mask_aug.dtype)
            mask_aug = mask_aug * keep

        # ---- part-level dropout ----
        p_part = self.clinical_part_mask_drop
        if p_part > 0.0 and hasattr(self, "part_slices"):
            # 依赖 part_slices 的 key 顺序保持一致
            part_names = list(self.part_slices.keys())
            B = clin.shape[0]
            P = len(part_names)
            part_keep = (torch.rand(B, P, device=clin.device) > p_part).to(mask_aug.dtype)  # 1 keep, 0 drop
            for j, pname in enumerate(part_names):
                sl = self.part_slices[pname]
                mask_aug[:, sl] = mask_aug[:, sl] * part_keep[:, j].unsqueeze(1)

        # 将被drop/缺失的特征值置0，避免“值残留”影响后续网络
        clin_aug = clin * (mask_aug >= 0.5).to(clin.dtype)

        return clin_aug, mask_aug

    def forward(
        self,
        x: torch.Tensor,
        clin: Optional[torch.Tensor] = None,
        clin_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        # ---- image branch ----
        if x.dim() == 2:
            h = self.embed_img(x)                      # (N, embed_dim)
            h = self._apply_instance_dropout(h)
            a_patch = self.attn(h)                     # (N,)
            z_img = torch.sum(a_patch.unsqueeze(-1) * h, dim=0).unsqueeze(0)  # (1, embed_dim)
        else:
            h = self.embed_img(x)                      # (B, N, embed_dim)
            h = self._apply_instance_dropout(h)
            a_patch = self.attn(h)                     # (B, N)
            z_img = torch.sum(a_patch.unsqueeze(-1) * h, dim=1)              # (B, embed_dim)

        if not self.use_clinical:
            logits = self.classifier(z_img)
            return (logits, a_patch) if return_attention else logits

        # ---- clinical branch ----
        if clin is None:
            raise ValueError("use_clinical=True 时，必须提供 clin (B,15) 或 (15,)")

        if clin_mask is None:
            clin_mask = torch.ones_like(clin, dtype=clin.dtype, device=clin.device)

        if clin.dim() == 1:
            clin = clin.unsqueeze(0)
            clin_mask = clin_mask.unsqueeze(0)

        # ===== NEW: train-time clinical mask augmentation =====
        clin_aug, mask_aug = self._augment_clinical_masks(clin, clin_mask)

        # 图像条件化 gating：用 z_img 生成 15维 gate
        clin_gated, feat_gate = self.clin_gating(z_img, clin_aug, mask_aug)

        # 分组临床编码（仍使用 mask）
        z_clin, part_gates = self.clin_enc(clin_gated, mask_aug, self.part_slices)


        # 融合：保持你原来 concat 的风格（你未要求改融合）
        z = torch.cat([z_img, z_clin], dim=1)
        logits = self.classifier(z)

        if return_attention:
            return logits, a_patch, feat_gate, part_gates

        return logits


# =====================================================
# 8. 训练/推理侧便捷接口
# =====================================================
def attach_clinical_to_npy(
    npy_path: str,
    clinical_table: ClinicalTable,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[str], torch.Tensor, torch.Tensor]:
    pid, clin_np, mask_np = clinical_table.get_by_npy_name(npy_path)
    clin_t = torch.from_numpy(clin_np).float()
    mask_t = torch.from_numpy(mask_np).float()
    if device is not None:
        clin_t = clin_t.to(device)
        mask_t = mask_t.to(device)
    return pid, clin_t, mask_t


if __name__ == "__main__":
    # 这里不做任何fit/标准化
    # clinical_csv = r"D:\PHD_project\Tumors\data\clinical.csv"
    # ct = ClinicalTable(clinical_csv, pid_len=7, standardize_continuous=False)
    # pid, clin, mask = ct.get_by_npy_name(r"D:\...\some_slide_1234567_feats.npy")
    # print(pid, clin, mask)
    pass
