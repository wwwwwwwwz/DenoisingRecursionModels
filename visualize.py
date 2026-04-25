import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import math
import random

# -------------------------
# ARC colormap
# -------------------------
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 cyan
    "#870C25",  # 9 brown
]

cmap = ListedColormap(ARC_COLORS)
norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)


def to_grid(tensor):
    t = tensor.squeeze()

    if t.ndim == 1:
        n = t.numel()
        side = int(math.sqrt(n))
        assert side * side == n, f"Cannot reshape length {n}"
        return t.view(side, side).numpy()

    if t.ndim == 2:
        return t.numpy()

    raise ValueError(f"Unsupported tensor shape {t.shape}")


def plot_grid(ax, grid, title):
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def visualize_example(pt_path, out_dir, prefix):
    data = torch.load(pt_path, map_location="cpu")

    panels = []

    if "inputs" in data:
        panels.append(("Input", to_grid(data["inputs"])))

    pred_keys = sorted(
        [k for k in data.keys() if k.startswith("prediction_")],
        key=lambda x: int(x.split("_")[1])
    )

    for k in pred_keys:
        step = k.split("_")[1]
        panels.append((f"Pred {step}", to_grid(data[k])))

    if "labels" in data:
        panels.append(("Label", to_grid(data["labels"])))

    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, (title, grid) in zip(axes, panels):
        plot_grid(ax, grid, title)

    base = os.path.splitext(os.path.basename(pt_path))[0]
    out_name = f"{prefix}_{base}.png"
    out_path = os.path.join(out_dir, out_name)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Saved {out_path}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    VIS_ROOT = "./vis_eval"
    OUT_DIR = os.path.join(VIS_ROOT, "images")
    MAX_PER_BATCH = 5

    os.makedirs(OUT_DIR, exist_ok=True)

    for d in sorted(os.listdir(VIS_ROOT)):
        subdir = os.path.join(VIS_ROOT, d)

        if not os.path.isdir(subdir):
            continue
        if d == "images":
            continue

        pt_files = sorted(f for f in os.listdir(subdir) if f.endswith(".pt"))
        if not pt_files:
            continue

        # ✅ TAKE AT MOST 5
        #pt_files = pt_files[:MAX_PER_BATCH]
        pt_files = random.sample(pt_files, k=min(MAX_PER_BATCH, len(pt_files)))

        print(f"\nProcessing {d} ({len(pt_files)} files)")

        for f in pt_files:
            visualize_example(
                pt_path=os.path.join(subdir, f),
                out_dir=OUT_DIR,
                prefix=d
            )
