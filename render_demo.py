"""

Render a 3D skeleton animation of one sequence with live GT vs HiST-VQ labels,

a ground-anchored camera, and a segmentation strip with a moving time cursor.



- Inference uses the preprocessed (hip-normalized) features the model expects.

- Rendering uses RAW world coordinates from the LARa CSV (downsampled to match),

  so the figure stands on the ground instead of being pinned at the root.

- A bottom strip shows the full-window GT and HiST-VQ segmentation with a cursor.



Outputs a GIF (no ffmpeg in this sandbox).

"""

import os

import argparse

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter



import torch



from src.model.histvq import HiSTVQModel

from src.model.eval_utils import read_mapping_file, create_correspondences, bounds



# Real LARa 22-joint topology (parent per joint), matching the OMoCap CSV order.

LARA_PARENT = [11, 0, 11, 6, 21, 7, 2, 4, 5, 3, 9, 21,

               11, 16, 21, 17, 12, 14, 15, 13, 19, -1]

LARA_BONES = [(i, p) for i, p in enumerate(LARA_PARENT) if p >= 0]





def load_world_xyz(raw_csv, sample_rate=4):

    """Raw (un-normalized) world joint positions from a LARa CSV -> (T,22,3)."""

    df = pd.read_csv(raw_csv, skiprows=4)

    df = df.drop(df.columns[[0, 1]], axis=1).iloc[::sample_rate]

    arr = df.to_numpy().reshape(df.shape[0], 22, 6)

    return arr[:, :, 3:6].astype(np.float64)  # TX,TY,TZ





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--vid", default="L01_S01_R02")

    ap.add_argument("--ckpt", default="models/pretrained/lara.model")

    ap.add_argument("--num_actions", type=int, default=8)

    ap.add_argument("--data", default="data/lara")

    ap.add_argument("--raw_csv", default=None, help="raw LARa CSV for ground-anchored world coords")

    ap.add_argument("--out", default="vis/skeleton_demo.gif")

    ap.add_argument("--start", type=int, default=0)

    ap.add_argument("--end", type=int, default=-1)

    ap.add_argument("--step", type=int, default=5)

    ap.add_argument("--fps", type=int, default=15)

    args = ap.parse_args()



    BONES = LARA_BONES

    device = torch.device("cpu")



    feat = np.load(os.path.join(args.data, "features", args.vid + ".npy"))  # (6,T,22,1)

    T = feat.shape[1]

    gt_names = open(os.path.join(args.data, "groundTruth", args.vid + ".txt")).read().splitlines()

    id2name = read_mapping_file(os.path.join(args.data, "mapping", "mapping.txt"))

    name2id = {v: k for k, v in id2name.items()}

    gt = np.array([name2id[n] for n in gt_names])



    # World coords for rendering (ground-anchored); fall back to normalized feat.

    if args.raw_csv:

        world = load_world_xyz(args.raw_csv)[:T]

    else:

        world = np.transpose(feat[3:6, :, :, 0], (1, 2, 0))

    # axes: x lateral, y depth, z up -> matplotlib (x, y, z)



    # Inference on full sequence

    model = HiSTVQModel(in_channels=6, filters=128, num_layers=3, latent_dim=16,

                        num_actions=args.num_actions, num_joints=22, num_person=1,

                        patch_size=50)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    model.to(device).eval()

    with torch.no_grad():

        x = torch.tensor(feat, dtype=torch.float).unsqueeze(0).to(device)

        model(x, torch.ones_like(x))

        pred_raw = model.indices.squeeze(0).cpu().numpy()

    pred = create_correspondences(gt, pred_raw)

    print(f"{args.vid}: T={T}  MoF(full)={100.0*np.mean(gt==pred):.1f}%")



    ids = sorted(set(gt.tolist()) | set(int(p) for p in pred.tolist()))

    cmap = plt.get_cmap("tab10")

    color = {a: cmap(i % 10) for i, a in enumerate(ids)}



    end = T if args.end < 0 else min(args.end, T)

    frames = list(range(args.start, end, args.step))



    # Ground-anchored camera: z (ground) is absolute so feet stay planted;

    # the camera follows the figure horizontally so it stays framed at a

    # constant scale (the subject walks several meters across the room).

    w = world[args.start:end]

    zmin = w.reshape(-1, 3)[:, 2].min()

    # body size -> constant render scale

    body_h = np.percentile(w[:, :, 2].max(1) - w[:, :, 2].min(1), 90)

    span = max(body_h * 0.65, 600.0)



    fig = plt.figure(figsize=(7.5, 9))
