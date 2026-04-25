
#!/usr/bin/env python3

import argparse
import os
import sys
from typing import List, Optional

import torch
import torch.distributed as dist
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from pretrain import (
    PretrainConfig,
    TrainState,
    create_dataloader,
    create_evaluators,
    create_model,
    evaluate,
    load_synced_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation for a checkpoint using the same pipeline as training."
    )
    parser.add_argument(
        "--config-path",
        default="configs",
        help="Directory that contains the Hydra config (default: %(default)s).",
    )
    parser.add_argument(
        "--config-name",
        default="cfg_pretrain",
        help="Config name to compose (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate. Use 'train' to evaluate on the training split (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path to load. Overrides config.load_checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saving evaluation artifacts (metrics, submissions, predictions). "
        "Overrides config.checkpoint_path. Defaults to the config value.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Override the step number associated with the checkpoint for logging and artifact naming.",
    )
    parser.add_argument(
        "--skip-evaluators",
        action="store_true",
        help="Skip custom evaluators (e.g., ARC submission generation).",
    )

    known_args, hydra_overrides = parser.parse_known_args()
    setattr(known_args, "hydra_overrides", hydra_overrides)
    return known_args


def _compose_config(args: argparse.Namespace) -> DictConfig:
    with initialize(config_path=args.config_path, version_base=None):
        return compose(config_name=args.config_name, overrides=args.hydra_overrides)


def _ensure_distributed() -> tuple[int, int, Optional[dist.ProcessGroup]]:
    rank = 0
    world_size = 1
    cpu_group = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        cpu_group = dist.new_group(backend="gloo")
        assert dist.get_rank(cpu_group) == rank and dist.get_world_size(cpu_group) == world_size

    return rank, world_size, cpu_group


def _infer_step(checkpoint_path: str) -> Optional[int]:
    base = os.path.basename(checkpoint_path.rstrip(os.sep))
    if base.startswith("step_"):
        try:
            return int(base.split("_")[-1])
        except ValueError:
            return None
    return None


def main() -> int:
    args = _parse_args()
    hydra_config = _compose_config(args)

    rank, world_size, cpu_group = _ensure_distributed()

    config = load_synced_config(hydra_config, rank=rank, world_size=world_size)

    if args.checkpoint is not None:
        config.load_checkpoint = args.checkpoint
    if config.load_checkpoint is None:
        raise ValueError("No checkpoint provided. Use --checkpoint or set load_checkpoint=<path> in the config overrides.")

    if args.output_dir is not None:
        config.checkpoint_path = args.output_dir
    elif config.checkpoint_path is None:
        ckpt_name = os.path.splitext(os.path.basename(config.load_checkpoint))[0]
        config.checkpoint_path = os.path.join("checkpoints", "eval", ckpt_name)

    os.makedirs(config.checkpoint_path, exist_ok=True)

    torch.random.manual_seed(config.seed + rank)

    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            split=args.split,
            rank=rank,
            world_size=world_size,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to create dataloader for split='{args.split}': {exc}") from exc

    # Build model shapes from train metadata, matching training-time create_model behavior.
    # Fallback to eval metadata when train split is unavailable.
    model_metadata = eval_metadata
    try:
        _train_loader, train_metadata = create_dataloader(
            config,
            split="train",
            rank=rank,
            world_size=world_size,
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
        )
        model_metadata = train_metadata
        del _train_loader
    except Exception as exc:
        if rank == 0:
            print(
                "Warning: could not load train metadata for model construction; "
                f"falling back to eval metadata ({exc})."
            )

    evaluators: List[object] = []
    if not args.skip_evaluators:
        try:
            evaluators = create_evaluators(config, eval_metadata)
        except Exception as exc:
            if rank == 0:
                print(f"Warning: failed to instantiate evaluators ({exc}); continuing without them.")

    model, _, _ = create_model(config, model_metadata, rank=rank, world_size=world_size)
    model.eval()

    step = args.step
    if step is None:
        step = _infer_step(config.load_checkpoint) or 0

    train_state = TrainState(
        model=model,
        optimizers=[],
        optimizer_lrs=[],
        carry=None,
        step=step,
        total_steps=step,
    )

    metrics = evaluate(
        config=config,
        train_state=train_state,
        eval_loader=eval_loader,
        eval_metadata=eval_metadata,
        evaluators=evaluators,
        rank=rank,
        world_size=world_size,
        cpu_group=cpu_group,
    )

    if rank == 0:
        if metrics:
            print("Evaluation metrics:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        print(f"  {key}/{inner_key}: {inner_value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("Evaluation completed but no metrics were produced.")

    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
