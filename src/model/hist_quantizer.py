# =============================================================================
# hist_quantizer.py — Hierarchical two-level patch VQ for HiST-VQ
# Built on SMQ's SkeletonMotionQuantizer (single-level). See reproduction spec
# sections 4 (two-level quantization), 5 (EMA + dead-code reset), 7.1 (dual
# commitment). Level 1 = sub-actions Z (alpha*K codes), Level 2 = actions A
# (K codes). Level-2 input is the level-1 codeword (cascaded quantization).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from src.model.motion_quantizer import euclidean_dist, laplace_smoothing


class HierarchicalMotionQuantizer(nn.Module):
    """Cascaded patch VQ: patch p -> sub-action z -> action a.

    Codebooks are EMA-updated (no codebook gradient); the encoder is trained via
    straight-through + two commitment losses. Returns depatchified Q_Z (for the
    temporal decoder), Q_A (for the spatial decoder), patch-grid Q_Z (per-patch
    sub-action codes), per-patch action indices (segmentation), a patch-validity
    mask, and the two commitment losses.
    """

    def __init__(self, embedding_dim, window, num_actions, alpha=2,
                 decay=0.5, eps=1e-5, nu_z=3, nu_a=1):
        super().__init__()
        self.E = embedding_dim                 # V * D (joints * latent)
        self.W = window                        # patch size P
        self.Ka = num_actions                  # second-level codebook size K
        self.Kz = alpha * num_actions          # first-level codebook size alpha*K
        self.decay = decay
        self.eps = eps
        self.nu_z = nu_z
        self.nu_a = nu_a

        # Codebooks as buffers -> never touched by the optimizer (EMA only).
        self.register_buffer("Z", torch.zeros(self.Kz, window, embedding_dim))
        self.register_buffer("A", torch.zeros(self.Ka, window, embedding_dim))
        self.register_buffer("initted", torch.tensor(False))

        # EMA running statistics (cluster size + running sum) per level.
        self.register_buffer("z_size", torch.zeros(self.Kz, 1, 1))
        self.register_buffer("z_avg", torch.zeros(self.Kz, window, embedding_dim))
        self.register_buffer("a_size", torch.zeros(self.Ka, 1, 1))
        self.register_buffer("a_avg", torch.zeros(self.Ka, window, embedding_dim))

    # ----------------------------------------------------------------- init
    def _init_codebooks(self, patches):
        if bool(self.initted):
            return
        zi = torch.empty_like(self.Z)
        ai = torch.empty_like(self.A)
        nn.init.kaiming_uniform_(zi)
        nn.init.kaiming_uniform_(ai)
        self.Z.data.copy_(zi)
        self.A.data.copy_(ai)
        self.z_avg.data.copy_(zi)
        self.a_avg.data.copy_(ai)
        self.z_size.data.fill_(1)
        self.a_size.data.fill_(1)
        self.initted.data.copy_(torch.tensor(True))

    # ------------------------------------------------------------- EMA step
    def _ema(self, codebook, size, avg, samples, onehot, K):
        """One EMA update for a level. samples:(Nv,W,E) onehot:(Nv,K)."""
        size.data.lerp_(onehot.sum(0).unsqueeze(1).unsqueeze(-1), 1 - self.decay)
        embed_sum = einsum("nwe,nk->kwe", samples, onehot)        # (K,W,E)
        avg.data.lerp_(embed_sum, 1 - self.decay)
        cs = laplace_smoothing(size, K) * size.sum(dim=-1, keepdim=True)
        codebook.data.copy_(avg / cs)

    def _revive(self, codebook, size, avg, samples, nu):
        """Replace dead codes (size < nu) with random current-batch samples."""
        dead = (size.squeeze() < nu).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0 or samples.shape[0] == 0:
            return
        pick = torch.randint(0, samples.shape[0], (dead.numel(),), device=samples.device)
        chosen = samples[pick]                                    # (n_dead,W,E)
        codebook.data[dead] = chosen
        avg.data[dead] = chosen
        size.data[dead] = float(nu)

    # ------------------------------------------------------------- forward
    def forward(self, x, mask):
        """x:(B,T,E) mask:(B,T,E) -> dict of outputs (see class docstring)."""
        B, T, E = x.shape
        W = self.W

        # Pad time so T is a multiple of W, then patchify to (B,P,W,E).
        rem = T % W
        pad = (W - rem) if rem != 0 else 0
        xp = F.pad(x, (0, 0, 0, pad))
        mp = F.pad(mask, (0, 0, 0, pad))
        Tp = T + pad
        P = Tp // W

        x_patch = xp.reshape(B * P, W, E)                         # (B*P,W,E)
        m_patch = mp.reshape(B * P, W, E)
        valid = m_patch.sum(dim=(1, 2)) > 0                       # (B*P,)
        vp = x_patch[valid]                                       # (Nv,W,E)

        self._init_codebooks(vp)

        # ---- Level 1: patch -> sub-action ----
        dZ = -euclidean_dist(vp, self.Z)                         # (Nv,Kz)
        idxZ = dZ.argmax(dim=1)                                   # (Nv,)
        oneZ = F.one_hot(idxZ, self.Kz).float()
        qZ = self.Z[idxZ]                                         # (Nv,W,E) buffer (no grad)
        qZ_st = vp + (qZ - vp).detach()                          # straight-through

        # ---- Level 2: sub-action -> action (input = qZ) ----
        dA = -euclidean_dist(qZ, self.A)                        # (Nv,Ka)
        idxA = dA.argmax(dim=1)                                   # (Nv,)
        oneA = F.one_hot(idxA, self.Ka).float()
        qA = self.A[idxA]                                         # (Nv,W,E)
        qA_st = qZ_st + (qA - qZ_st).detach()                    # straight-through

        # ---- dual commitment (encoder-side only) ----
        commitZ = F.mse_loss(vp, qZ.detach())
        commitA = F.mse_loss(qZ_st, qA.detach())

        # ---- EMA codebook updates + dead-code revival ----
        if self.training:
            self._ema(self.Z, self.z_size, self.z_avg, vp, oneZ, self.Kz)
            self._ema(self.A, self.a_size, self.a_avg, qZ, oneA, self.Ka)
            self._revive(self.Z, self.z_size, self.z_avg, vp, self.nu_z)
            self._revive(self.A, self.a_size, self.a_avg, qZ, self.nu_a)

        # ---- scatter valid patches back to full (B*P) grid ----
        def scatter(vals, fill_shape):
            out = torch.zeros(fill_shape, dtype=vals.dtype, device=vals.device)
            out[valid] = vals
            return out

        qZ_full = scatter(qZ_st, (B * P, W, E))                  # (B*P,W,E)
        qA_full = scatter(qA_st, (B * P, W, E))
        idxA_full = scatter(idxA, (B * P,))                      # (B*P,)

        # depatchify -> (B,T,E), cropped back to original length
        end = -pad if pad > 0 else None
        Q_Z = qZ_full.reshape(B, Tp, E)[:, :end, :].contiguous()
        Q_A = qA_full.reshape(B, Tp, E)[:, :end, :].contiguous()

        # patch-grid views for the temporal decoder / segmentation
        qZ_grid = qZ_full.reshape(B, P, W * E)                   # (B,P,W*E)
        idxA_grid = idxA_full.reshape(B, P)                      # (B,P)
        patch_valid = valid.reshape(B, P)                        # (B,P) bool

        # per-frame action indices (segmentation), cropped to T
        frame_idx = idxA_grid.repeat_interleave(W, dim=1)[:, :end]  # (B,T)

        return {
            "Q_Z": Q_Z, "Q_A": Q_A,
            "qZ_grid": qZ_grid, "idxA_grid": idxA_grid,
            "patch_valid": patch_valid, "frame_idx": frame_idx,
            "commitZ": commitZ, "commitA": commitA,
        }
