from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from src.model.histvq import HiSTVQModel
from src.model.utils import distance_joints
from src.model.eval_utils import evaluate_local_hungarian, evaluate_global_hungarian


def make_timestamps(patch_valid):
    """patch_valid:(B,P) bool -> (tau, mask) patch-level GT timestamps in [0,1].

    Per sequence, valid patches are linearly mapped to [0,1] (CTE-style relative
    timestamps); padded patches get 0 and are masked out of the temporal loss.
    """
    B, P = patch_valid.shape
    tau = torch.zeros(B, P, device=patch_valid.device)
    for n in range(B):
        idx = patch_valid[n].nonzero(as_tuple=False).flatten()
        mv = idx.numel()
        if mv > 1:
            tau[n, idx] = torch.linspace(0.0, 1.0, mv, device=patch_valid.device)
    return tau, patch_valid.float()


class Trainer:
    """Trains HiST-VQ and evaluates with MoF, Edit and F1 (SMQ protocol)."""

    def __init__(self, in_channels, filters, num_layers, latent_dim, num_actions,
                 num_joints, num_person, patch_size, alpha=2, decay=0.5,
                 nu_z=3, nu_a=1):
        self.model = HiSTVQModel(
            in_channels=in_channels, filters=filters, num_layers=num_layers,
            latent_dim=latent_dim, num_actions=num_actions, num_joints=num_joints,
            num_person=num_person, patch_size=patch_size, alpha=alpha, decay=decay,
            nu_z=nu_z, nu_a=nu_a)
        self.mse = nn.MSELoss(reduction="none")

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate,
              lambda_commit, lambda_spat, lambda_temp, device,
              joint_distance_recons=True):
        self.model.train()
        self.model.to(device)
        num_batches = batch_gen.num_batches(batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            pbar = tqdm(total=num_batches, desc=f"Training [Epoch {epoch+1}]",
                        unit="batch", leave=False)
            ep_spat = ep_temp = ep_cz = ep_ca = 0.0

            while batch_gen.has_next():
                batch_input, mask = batch_gen.next_batch(batch_size)
                batch_input, mask = batch_input.to(device), mask.to(device)

                optimizer.zero_grad()
                S_hat, T_hat = self.model(batch_input, mask)

                # spatial reconstruction (inter-joint distance MSE) [reuse-SMQ]
                if joint_distance_recons:
                    x, x_hat = distance_joints(batch_input), distance_joints(S_hat)
                else:
                    x, x_hat = batch_input, S_hat
                L_spat = torch.mean(self.mse(x, x_hat))

                # temporal reconstruction (patch-level timestamp MSE) [new-HiST]
                tau, pmask = make_timestamps(self.model.patch_valid)
                L_temp = (((T_hat - tau) ** 2) * pmask).sum() / pmask.sum().clamp(min=1)

                L_cz, L_ca = self.model.commitZ, self.model.commitA
                loss = (lambda_commit * (L_cz + L_ca)
                        + lambda_spat * L_spat
                        + lambda_temp * L_temp)

                loss.backward()
                optimizer.step()

                ep_spat += L_spat.item(); ep_temp += L_temp.item()
                ep_cz += L_cz.item(); ep_ca += L_ca.item()
                pbar.update(1)

            batch_gen.reset()
            pbar.close()

            if (epoch + 1) % 5 == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_dir / f"epoch-{epoch+1}.model")

            print("[epoch %d]: Spat=%.5f Temp=%.5f CommitZ=%.5f CommitA=%.5f" %
                  (epoch + 1, ep_spat / num_batches, ep_temp / num_batches,
                   ep_cz / num_batches, ep_ca / num_batches))

    def eval(self, model_path, features_path, gt_path, mapping_file, epoch, vis,
             plot_dir, device):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))

            local_mof, local_edit, local_f1, gt_all, pred_all = evaluate_local_hungarian(
                model=self.model, features_path=features_path, gt_path=gt_path,
                mapping_file=mapping_file, epoch=epoch, device=device, verbose=True)

            evaluate_global_hungarian(
                model=self.model, features_path=features_path, gt_path=gt_path,
                device=device, mapping_file=mapping_file, epoch=epoch, vis=vis,
                plot_dir=plot_dir, gt_all=gt_all, prediction_all=pred_all, verbose=True)
