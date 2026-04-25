import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import math
from PIL import Image
import io

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


def render_frame(grid, title):
    """
    Render a single ARC grid frame to a PIL Image.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def visualize_example_gif(pt_path, out_dir, prefix, frame_duration=600):
    data = torch.load(pt_path, map_location="cpu")

    frames = []

    # -------------------------
    # Input
    # -------------------------
    if "inputs" in data:
        frames.append(
            render_frame(to_grid(data["inputs"]), "Input")
        )

    # -------------------------
    # Predictions (ordered)
    # -------------------------
    pred_keys = sorted(
        [k for k in data.keys() if k.startswith("prediction_")],
        key=lambda x: int(x.split("_")[1])
    )

    for k in pred_keys:
        step = k.split("_")[1]

        q_key = f"q_{step}"
        if q_key in data:
            q_val = data[q_key].item()
            title = f"Prediction {step} | q_={q_val:.3f}"
        else:
            title = f"Prediction {step}"

        frames.append(
            render_frame(to_grid(data[k]), title)
        )

    # -------------------------
    # Label
    # -------------------------
    if "labels" in data:
        frames.append(
            render_frame(to_grid(data["labels"]), "Label")
        )

    if not frames:
        return

    base = os.path.splitext(os.path.basename(pt_path))[0]
    out_name = f"{prefix}_{base}.gif"
    out_path = os.path.join(out_dir, out_name)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,  # ms per frame
        loop=0,
        optimize=True,
    )

    print(f"[✓] Saved {out_path}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    # import random

    # VIS_ROOT = "./vis_eval_2"
    # OUT_DIR = os.path.join(VIS_ROOT, "gifs")  # or "images"
    # MAX_PER_BATCH = 5

    # os.makedirs(OUT_DIR, exist_ok=True)

    # for d in sorted(os.listdir(VIS_ROOT)):
    #     subdir = os.path.join(VIS_ROOT, d)

    #     # Skip non-directories and output dir
    #     if not os.path.isdir(subdir):
    #         continue
    #     if d == os.path.basename(OUT_DIR):
    #         continue

    #     pt_files = sorted(
    #         f for f in os.listdir(subdir) if f.endswith(".pt")
    #     )
    #     if not pt_files:
    #         continue

    #     # Sample at most MAX_PER_BATCH
    #     pt_files = random.sample(
    #         pt_files, k=min(MAX_PER_BATCH, len(pt_files))
    #     )

    #     print(f"\nProcessing {d} ({len(pt_files)} files)")

    #     for f in pt_files:
    #         visualize_example_gif(
    #             pt_path=os.path.join(subdir, f),
    #             out_dir=OUT_DIR,
    #             prefix=d,
    #         )

    PT_PATH = "./vis_eval_2/visualization_60/example_6.pt" 
    OUT_DIR = "./vis_eval/gifs" 
    os.makedirs(OUT_DIR, exist_ok=True) 
    visualize_example_gif( pt_path=PT_PATH, out_dir=OUT_DIR, prefix="single" )
