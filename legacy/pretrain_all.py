#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Dict, List, Tuple


MODEL_SCRIPT_MAP: Dict[Tuple[str, bool], str] = {
    ("trm", False): "pretrain.py",
    ("trm", True): "pretrain_visualize.py",
    ("sprm", False): "pretrain_sprm.py",
    ("sprm", True): "pretrain_sprm_visualize.py",
    ("drm", False): "pretrain_drm.py",
    ("drm", True): "pretrain_drm_visualize.py",
}

MODEL_ARCH_MAP: Dict[str, str] = {
    "trm": "trm",
    "sprm": "trm_sprm",
    "drm": "trm_drm",
}


def has_arch_override(extra_args: List[str]) -> bool:
    arch_prefixes = (
        "arch=",
        "+arch=",
        "++arch=",
        "arch.name=",
        "+arch.name=",
        "++arch.name=",
    )
    return any(arg.startswith(arch_prefixes) for arg in extra_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unified launcher for TRM/DRM/SPRM pretraining scripts. "
            "All unknown args are forwarded to Hydra."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model",
        choices=("trm", "sprm", "drm"),
        default="trm",
        help="Select the training variant to run.",
    )
    parser.add_argument(
        "--vis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use visualization-enabled script variant.",
    )

    args, hydra_args = parser.parse_known_args()

    target_script = MODEL_SCRIPT_MAP[(args.model, args.vis)]
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), target_script)

    if not os.path.exists(target_path):
        print(f"Target script not found: {target_path}", file=sys.stderr)
        return 1

    # Keep behavior consistent with README defaults by setting arch unless user overrides it.
    if not has_arch_override(hydra_args):
        hydra_args = [f"arch={MODEL_ARCH_MAP[args.model]}"] + hydra_args

    python_exe = sys.executable or "python"
    argv = [python_exe, target_path] + hydra_args

    os.execv(python_exe, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
