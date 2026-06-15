# =============================================================================
# histvq.py — HiST-VQ model: joint-decoupled MS-TCN encoder, two-level
# hierarchical VQ, spatial decoder (Q_A -> skeleton) and temporal decoder
# (Q_Z -> patch-level timestamps). Built on SMQ's SMQModel.
# =============================================================================

import torch
import torch.nn as nn

from src.model.hist_quantizer import HierarchicalMotionQuantizer
from src.model.utils import process_mask
from src.model.ms_tcn import MultiStageModel


class TemporalDecoder(nn.Module):
    """2-hidden-layer MLP mapping a per-patch sub-action code (W*E) -> scalar timestamp."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):                       # x:(B,P,in_dim)
        return self.net(x).squeeze(-1)          # (B,P)


class HiSTVQModel(nn.Module):
    """TCN autoencoder with hierarchical spatiotemporal VQ.

    forward returns (S_hat, T_hat). Auxiliary quantities (commitments, per-frame
    action indices for segmentation, patch validity) are stored as attributes so
    the Trainer and the SMQ-style evaluator can read them.
    """

    def __init__(self, in_channels=3, filters=128, num_layers=3, latent_dim=16,
                 num_actions=8, num_joints=22, num_person=1, patch_size=50,
                 alpha=2, decay=0.5, nu_z=3, nu_a=1, temporal_hidden=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.num_person = num_person

        # Outputs filled in forward()
        self.commitZ = None
        self.commitA = None
        self.indices = None          # (B,T) per-frame action index (segmentation)
        self.patch_valid = None      # (B,P) bool

        emb_dim = latent_dim * num_joints * num_person   # E = V*M*D

        # Encoder (joint-decoupled) [reuse-SMQ]
        self.encoder = MultiStageModel(num_layers=num_layers, num_f_maps=filters,
                                       dim=in_channels, target_dim1=int(latent_dim / 2),
                                       target_dim2=latent_dim)

        # Hierarchical two-level VQ [new-HiST]
        self.vq = HierarchicalMotionQuantizer(embedding_dim=emb_dim, window=patch_size,
                                              num_actions=num_actions, alpha=alpha,
                                              decay=decay, nu_z=nu_z, nu_a=nu_a)

        # Spatial decoder (mirror of encoder), input = Q_A [reuse-SMQ arch]
        self.spatial_decoder = MultiStageModel(num_layers=num_layers, num_f_maps=filters,
                                               dim=latent_dim, target_dim1=int(latent_dim / 2),
                                               target_dim2=in_channels)

        # Temporal decoder MLP, input = per-patch Q_Z [new-HiST]
        self.temporal_decoder = TemporalDecoder(patch_size * emb_dim, hidden=temporal_hidden)

    def forward(self, x, mask):
        """x,mask:(N,C,T,V,M) -> S_hat:(N,C,T,V,M), T_hat:(N,P)."""
        N, C, T, V, M = x.size()

        # Pack person+joint into batch -> (N*M*V, C, T)  [reuse-SMQ]
        xp = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, C, T)
        mp = mask.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, C, T)

        latent = self.encoder(xp, mp)                                # (N*M*V, D, T)

        # Repack to per-frame skeleton token: (N, T, V*M*D)
        lat = latent.view(N * M, V, self.latent_dim, T).permute(0, 3, 1, 2)
        lat = lat.reshape(N, M, T, V, self.latent_dim).permute(0, 2, 3, 1, 4)
        lat = lat.reshape(N, T, -1)

        vq_mask = process_mask(mp, batch_size=N, num_person=M, num_joints=self.num_joints,
                               seq_length=T, num_features_per_joint=self.latent_dim)

        out = self.vq(lat, vq_mask)
        self.commitZ = out["commitZ"]
        self.commitA = out["commitA"]
        self.indices = out["frame_idx"]
        self.patch_valid = out["patch_valid"]

        # ---- spatial decode from Q_A ----  (N,T,E) -> (N*M*V, D, T)
        QA = out["Q_A"]
        QA = QA.reshape(N, T, V, M * self.latent_dim)
        QA = QA.reshape(N, T, V, M, self.latent_dim).permute(0, 3, 4, 1, 2)
        QA = QA.reshape(N * M, self.latent_dim, T, V).permute(0, 3, 1, 2)
        QA = QA.reshape(N * M * V, self.latent_dim, T)
        dec = self.spatial_decoder(QA, mp)                           # (N*M*V, C, T)
        dec = dec.reshape(N * M, V, C, T).reshape(N, M, V, C, T)
        S_hat = dec.permute(0, 3, 4, 2, 1)                           # (N,C,T,V,M)

        # ---- temporal decode from per-patch Q_Z ----
        T_hat = self.temporal_decoder(out["qZ_grid"])               # (N,P)

        return S_hat, T_hat
