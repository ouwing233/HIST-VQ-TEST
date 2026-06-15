from pathlib import Path
import argparse
import random

import torch

from src.model.utils import get_num_actions, print_run_summary
from batch_gen import BatchGenerator
from model import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# -------------------------------
# Dataset Defaults (P = 1 s patch)
# -------------------------------

def get_dataset_defaults(name: str):
    if name == "hugadb":
        return dict(batch_size=8,  num_features=6, num_joints=6,  num_person=1,
                    patch_size=60, lambda_temp=0.2)
    if name == "lara":
        return dict(batch_size=8,  num_features=6, num_joints=22, num_person=1,
                    patch_size=50, lambda_temp=0.1)
    if name in {"babel1", "babel2", "babel3"}:
        return dict(batch_size=32, num_features=3, num_joints=25, num_person=1,
                    patch_size=30, lambda_temp=0.02)
    raise ValueError(f"Unknown dataset: {name}")

# -------------------------------
# CLI
# -------------------------------

parser = argparse.ArgumentParser(description="Train and eval pipeline for HiST-VQ")

parser.add_argument("--action", choices=["train", "eval"], default="train")
parser.add_argument("--dataset", required=True, choices=["hugadb", "lara", "babel1", "babel2", "babel3"])
parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint for eval.")

# Training & model
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--num_f_maps", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--latent_dim", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--sample_rate", type=int, default=1)

# Hierarchical VQ
parser.add_argument("--patch_size", type=int)
parser.add_argument("--num_actions", type=int, help="K (= second-level codebook size).")
parser.add_argument("--alpha", type=int, default=2, help="First-level codebook = alpha*K.")
parser.add_argument("--decay", type=float, default=0.5, help="EMA decay beta.")
parser.add_argument("--nu_z", type=int, default=3, help="Dead-code threshold (sub-action).")
parser.add_argument("--nu_a", type=int, default=1, help="Dead-code threshold (action).")

# Loss weights
parser.add_argument("--lambda_commit", type=float, default=1.0)
parser.add_argument("--lambda_spat", type=float, default=0.001)
parser.add_argument("--lambda_temp", type=float, default=None, help="Default per dataset.")
parser.add_argument("--joint_distance_recons", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--vis", action="store_true")

# Paths
parser.add_argument("--data_root", type=Path, default=Path("./data"))
parser.add_argument("--models_root", type=Path, default=Path("./models"))
parser.add_argument("--vis_root", type=Path, default=Path("./vis"))

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    dataset_root = args.data_root / args.dataset
    features_path = dataset_root / "features"
    gt_path = dataset_root / "groundTruth"
    mapping_file = dataset_root / "mapping" / "mapping.txt"
    model_dir = args.models_root / args.dataset
    plot_dir = args.vis_root / args.dataset

    cfg = get_dataset_defaults(args.dataset)
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    patch_size = args.patch_size if args.patch_size is not None else cfg["patch_size"]
    lambda_temp = args.lambda_temp if args.lambda_temp is not None else cfg["lambda_temp"]
    num_features = cfg["num_features"]
    num_joints = cfg["num_joints"]
    num_person = cfg["num_person"]
    num_actions = args.num_actions if args.num_actions is not None else get_num_actions(gt_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    if args.vis:
        plot_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        in_channels=num_features, filters=args.num_f_maps, num_layers=args.num_layers,
        latent_dim=args.latent_dim, num_actions=num_actions, num_joints=num_joints,
        num_person=num_person, patch_size=patch_size, alpha=args.alpha,
        decay=args.decay, nu_z=args.nu_z, nu_a=args.nu_a)

    if args.action == "train":
        batch_gen = BatchGenerator(features_path=features_path, sample_rate=args.sample_rate,
                                   num_features=num_features, num_joints=num_joints,
                                   num_person=num_person)
        batch_gen.read_data()

        print_run_summary(dataset=args.dataset, num_features=num_features, num_joints=num_joints,
                          num_person=num_person, num_actions=num_actions, epochs=args.epoch,
                          batch_size=batch_size, learning_rate=args.lr, patch_size=patch_size)
        print(f"[HiST-VQ] alpha={args.alpha} -> Z codebook={args.alpha*num_actions}, A codebook={num_actions} | "
              f"lambda_commit={args.lambda_commit} lambda_spat={args.lambda_spat} lambda_temp={lambda_temp}")

        trainer.train(save_dir=model_dir, batch_gen=batch_gen, num_epochs=args.epoch,
                      batch_size=batch_size, learning_rate=args.lr,
                      lambda_commit=args.lambda_commit, lambda_spat=args.lambda_spat,
                      lambda_temp=lambda_temp, device=device,
                      joint_distance_recons=args.joint_distance_recons)

    elif args.action == "eval":
        ckpt_path = args.ckpt if args.ckpt is not None else model_dir / f"epoch-{args.epoch}.model"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        trainer.eval(model_path=ckpt_path, features_path=features_path, gt_path=gt_path,
                     mapping_file=mapping_file, epoch=args.epoch, vis=args.vis,
                     plot_dir=plot_dir, device=device)
