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
    gs = fig.add_gridspec(2, 1, height_ratios=[7, 1.2], hspace=0.12)
    ax = fig.add_subplot(gs[0], projection="3d")
    axs = fig.add_subplot(gs[1])

    # --- static segmentation strip (GT top row, SMQ bottom row) ---
    def draw_strip_row(y0, seq):
        for s, e, lab in bounds(list(seq)):
            axs.axvspan(s, e, y0, y0 + 0.5, facecolor=color[lab], alpha=1.0)
    draw_strip_row(0.5, gt)
    draw_strip_row(0.0, pred)
    axs.set_xlim(0, T)
    axs.set_ylim(0, 1)
    axs.set_yticks([0.25, 0.75])
    axs.set_yticklabels(["SMQ", "GT"], fontsize=10)
    axs.set_xlabel("frame")
    cursor = axs.axvline(frames[0], color="k", lw=1.5)
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[a]) for a in ids]
    axs.legend(handles, [id2name[a] for a in ids], loc="upper center",
               bbox_to_anchor=(0.5, -0.55), ncol=4, fontsize=8, frameon=False)

    def draw(fi):
        t = frames[fi]
        ax.cla()
        p = world[t]
        pcol = color.get(int(pred[t]), "gray")
        for i, par in BONES:
            ax.plot([p[i, 0], p[par, 0]], [p[i, 1], p[par, 1]], [p[i, 2], p[par, 2]],
                    "-o", ms=3, lw=2.5, color=pcol)
        # ground absolute (z), camera follows figure horizontally (x,y)
        rx, ry = p[21, 0], p[21, 1]
        ax.set_xlim(rx - span, rx + span)
        ax.set_ylim(ry - span, ry + span)
        ax.set_zlim(zmin, zmin + 2 * span)
        ax.set_box_aspect((1, 1, 2))
        ax.view_init(elev=12, azim=-60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        g, pr = id2name[int(gt[t])], id2name.get(int(pred[t]), "?")
        ok = "OK" if g == pr else "X"
        ax.set_title(f"frame {t}/{T}\nGT:  {g}\nSMQ: {pr}   [{ok}]",
                     fontsize=12, loc="left", color=("green" if g == pr else "red"))
        cursor.set_xdata([t, t])
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / args.fps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"Saved {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
