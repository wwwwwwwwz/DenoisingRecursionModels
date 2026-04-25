#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap


ARC_COLORS = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = BoundaryNorm(np.arange(-0.5, 10.5, 1), ARC_CMAP.N)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ARC DRM/TRM trajectory artifacts to a PDF report."
    )
    parser.add_argument(
        "input",
        help="Trajectory .json/.pt file or a directory containing saved diagnostics artifacts.",
    )
    parser.add_argument(
        "--output",
        default="arc_diagnostics.pdf",
        help="Output PDF path (default: %(default)s).",
    )
    parser.add_argument(
        "--export-json",
        default=None,
        help="Optional JSON export for drm_diagnostics_viewer_arc2.html.",
    )
    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="Only include trajectories that are correct at some step.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of trajectories rendered.",
    )
    return parser.parse_args()


def _to_grid(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        side = int(math.sqrt(arr.size))
        if side * side != arr.size:
            raise ValueError(f"Cannot reshape flat sequence of length {arr.size} into a square grid.")
        arr = arr.reshape(side, side)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported grid shape: {arr.shape}")
    return arr


def _tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _tensor_to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tensor_to_python(v) for v in value]
    return value


def _iter_trajectory_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and (
            path.parent.name == "trajectories"
            or path.name in {"selected.json", "debug.json", "all.json", "selected.pt", "debug.pt", "all.pt"}
        ):
            yield path


def _normalize_legacy_example(example: Dict[str, Any]) -> Dict[str, Any]:
    prediction_keys = sorted(
        [k for k in example.keys() if k.startswith("prediction_")],
        key=lambda key: int(key.split("_")[1]),
    )
    predictions = [example[key] for key in prediction_keys]
    q_values = []
    timesteps = []
    for key in prediction_keys:
        step = key.split("_")[1]
        q_values.append(example.get(f"q_{step}"))
        timesteps.append(None)

    normalized = dict(example)
    normalized.setdefault("predictions", predictions)
    normalized.setdefault("q_halt_probs", q_values)
    normalized.setdefault("timesteps", timesteps)
    normalized.setdefault("set_name", "unknown")
    normalized.setdefault("batch_index", -1)
    normalized.setdefault("batch_item_index", -1)
    normalized.setdefault("num_inference_steps", len(predictions))
    return normalized


def _load_trajectories(input_path: Path) -> Dict[str, Any]:
    all_trajectories: List[Dict[str, Any]] = []
    mode = "unknown"
    step = None

    for path in _iter_trajectory_files(input_path):
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "trajectories" in payload:
            all_trajectories.extend(payload["trajectories"])
            mode = payload.get("mode", mode)
            step = payload.get("step", step)
        elif isinstance(payload, dict):
            all_trajectories.append(_normalize_legacy_example(payload))
        else:
            raise ValueError(f"Unsupported payload format in {path}")

    if not all_trajectories:
        raise ValueError(f"No trajectory artifacts found in {input_path}")

    return {
        "mode": mode,
        "step": step,
        "trajectories": all_trajectories,
    }


def _build_panels(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    panels = [{"title": "Input", "grid": example["inputs"]}]

    q_values = example.get("q_halt_probs", [])
    timesteps = example.get("timesteps", [])
    for step_idx, pred in enumerate(example.get("predictions", []), start=1):
        title = f"Pred {step_idx}"
        extras = []
        if step_idx - 1 < len(timesteps) and timesteps[step_idx - 1] is not None:
            extras.append(f"t={timesteps[step_idx - 1]}")
        if step_idx - 1 < len(q_values) and q_values[step_idx - 1] is not None:
            q_value = q_values[step_idx - 1]
            if isinstance(q_value, torch.Tensor):
                q_value = q_value.item()
            extras.append(f"q={float(q_value):.3f}")
        if extras:
            title += " | " + ", ".join(extras)
        panels.append({"title": title, "grid": pred})

    panels.append({"title": "Label", "grid": example["labels"]})
    return panels


def _render_example(pdf: PdfPages, example: Dict[str, Any], index: int) -> None:
    panels = _build_panels(example)
    max_cols = 4
    cols = min(max_cols, len(panels))
    rows = math.ceil(len(panels) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, panel in zip(axes, panels):
        ax.imshow(_to_grid(panel["grid"]), cmap=ARC_CMAP, norm=ARC_NORM)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel["title"])

    for ax in axes[len(panels):]:
        ax.axis("off")

    metadata = (
        f"Example {index} | set={example.get('set_name')} | "
        f"batch={example.get('batch_index')} | item={example.get('batch_item_index')} | "
        f"final_correct={example.get('is_final_correct', False)} | "
        f"any_step_correct={example.get('is_any_step_correct', False)}"
    )
    fig.suptitle(metadata, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    payload = _load_trajectories(Path(args.input))
    trajectories = payload["trajectories"]

    if args.only_correct:
        trajectories = [item for item in trajectories if item.get("is_any_step_correct", False)]
    if args.max_examples is not None:
        trajectories = trajectories[:args.max_examples]

    if not trajectories:
        raise ValueError("No trajectories left after filtering.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with PdfPages(args.output) as pdf:
        for index, example in enumerate(trajectories, start=1):
            _render_example(pdf, example, index=index)

    if args.export_json is not None:
        export_payload = {
            "mode": payload.get("mode"),
            "step": payload.get("step"),
            "trajectories": [_tensor_to_python(example) for example in trajectories],
        }
        with open(args.export_json, "w", encoding="utf-8") as handle:
            json.dump(export_payload, handle)

    print(f"Saved PDF report to {args.output}")
    if args.export_json is not None:
        print(f"Saved JSON export to {args.export_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
