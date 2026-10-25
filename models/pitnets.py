"""Frozen PitNETs MIL architecture; checkpoint parameter names are preserved."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

def default_part_slices_for_15_features() -> Dict[str, slice]:

    return {
        "patient_info": slice(0, 2),
        "symptoms": slice(2, 8),
        "serology": slice(8, 14),
        "tumor_diameter": slice(14, 15),
    }

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

        if h.dim() == 2:
            v = self.attn_v(h)
            u = self.attn_u(h)
            a = self.attn_w(v * u).squeeze(-1)
            return torch.softmax(a, dim=0)
        elif h.dim() == 3:
            v = self.attn_v(h)
            u = self.attn_u(h)
            a = self.attn_w(v * u).squeeze(-1)
            return torch.softmax(a, dim=1)
        else:
            raise ValueError("Invalid input dim for GatedAttention")

class ImageConditionedClinicalGating(nn.Module):

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
        z_img: torch.Tensor,
        clin: torch.Tensor,
        clin_mask: torch.Tensor,
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

        gate = gate * (clin_mask >= 0.5).to(gate.dtype)

        clin_gated = clin * gate
        return clin_gated, gate

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
        num_layers: int = 1,
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

        tokens = torch.stack(tokens, dim=1)
        part_gates = torch.sigmoid(self.part_gate_logits)
        tokens = tokens * part_gates.view(1, -1, 1)

        tokens = self.cross_part(tokens)
        pooled = self.pool_norm(tokens.mean(dim=1))
        z_clin = self.fuse(pooled)

        if single:
            return z_clin.squeeze(0), part_gates
        return z_clin, part_gates

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

        attn_dim: int = 128,
        attn_dropout: Optional[float] = None,
        cls_hidden_dim: int = 256,
        cls_layers: int = 1,
        instance_dropout: float = 0.0,

        clin_dim: int = 15,
        gating_hidden: int = 128,
        gating_dropout: float = 0.1,

        clinical_feature_mask_drop: float = 0.0,
        clinical_part_mask_drop: float = 0.0,

        clin_part_hid: int = 64,
        clin_part_out: int = 64,
        clin_enc_dropout: float = 0.25,
        clin_n_heads: int = 2,
        clin_attn_ff: int = 128,
        clin_num_layers: int = 1,

    ):
        super().__init__()
        self.use_clinical = bool(use_clinical)

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

        self.clinical_feature_mask_drop = float(clinical_feature_mask_drop)
        self.clinical_part_mask_drop = float(clinical_part_mask_drop)

        if self.use_clinical:
            if part_slices is None:
                part_slices = default_part_slices_for_15_features()
            self.part_slices = part_slices

            if part_dims is None:
                part_dims = {k: (v.stop - v.start) for k, v in part_slices.items()}

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
        clin: torch.Tensor,
        clin_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.training:
            return clin, clin_mask

        mask_aug = clin_mask

        p_feat = self.clinical_feature_mask_drop
        if p_feat > 0.0:
            B, C = clin.shape
            keep = (torch.rand(B, C, device=clin.device) > p_feat).to(mask_aug.dtype)
            mask_aug = mask_aug * keep

        p_part = self.clinical_part_mask_drop
        if p_part > 0.0 and hasattr(self, "part_slices"):

            part_names = list(self.part_slices.keys())
            B = clin.shape[0]
            P = len(part_names)
            part_keep = (torch.rand(B, P, device=clin.device) > p_part).to(mask_aug.dtype)
            for j, pname in enumerate(part_names):
                sl = self.part_slices[pname]
                mask_aug[:, sl] = mask_aug[:, sl] * part_keep[:, j].unsqueeze(1)

        clin_aug = clin * (mask_aug >= 0.5).to(clin.dtype)

        return clin_aug, mask_aug

    def forward(
        self,
        x: torch.Tensor,
        clin: Optional[torch.Tensor] = None,
        clin_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):

        if x.dim() == 2:
            h = self.embed_img(x)
            h = self._apply_instance_dropout(h)
            a_patch = self.attn(h)
            z_img = torch.sum(a_patch.unsqueeze(-1) * h, dim=0).unsqueeze(0)
        else:
            h = self.embed_img(x)
            h = self._apply_instance_dropout(h)
            a_patch = self.attn(h)
            z_img = torch.sum(a_patch.unsqueeze(-1) * h, dim=1)

        if not self.use_clinical:
            logits = self.classifier(z_img)
            return (logits, a_patch) if return_attention else logits

        if clin is None:
            raise ValueError("Clinical inputs are required with shape (B,15) or (15,)")

        if clin_mask is None:
            clin_mask = torch.ones_like(clin, dtype=clin.dtype, device=clin.device)

        if clin.dim() == 1:
            clin = clin.unsqueeze(0)
            clin_mask = clin_mask.unsqueeze(0)

        clin_aug, mask_aug = self._augment_clinical_masks(clin, clin_mask)

        clin_gated, feat_gate = self.clin_gating(z_img, clin_aug, mask_aug)

        z_clin, part_gates = self.clin_enc(clin_gated, mask_aug, self.part_slices)

        z = torch.cat([z_img, z_clin], dim=1)
        logits = self.classifier(z)

        if return_attention:
            return logits, a_patch, feat_gate, part_gates

        return logits
