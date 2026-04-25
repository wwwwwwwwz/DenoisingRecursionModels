from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import types
import json

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class _FallbackDDPMScheduler:
    """Minimal DDPM scheduler used when diffusers is not installed."""
    def __init__(
        self,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        self.config = types.SimpleNamespace(num_train_timesteps=num_train_timesteps)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        alpha_t = alphas_cumprod[timesteps].view(-1, *([1] * (original_samples.ndim - 1)))
        return alpha_t.sqrt() * original_samples + (1.0 - alpha_t).sqrt() * noise


def _create_state_noise_scheduler():
    try:
        from diffusers import DDPMScheduler
    except ImportError:
        return _FallbackDDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
        )

    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="sample",
    )


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []
    eval_trajectory_mode: str = "off"
    max_tasks_per_set: Optional[int] = None

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        max_tasks_per_set=config.max_tasks_per_set,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def _format_set_list(set_names: Sequence[str], max_items: int = 4) -> str:
    if len(set_names) <= max_items:
        return ", ".join(set_names)
    visible = ", ".join(set_names[:max_items])
    return f"{visible}, ... ({len(set_names)} sets total)"


def _resolve_model_vocab_and_mask_token_id(
    config: PretrainConfig,
    metadata: PuzzleDatasetMetadata,
    *,
    rank: int,
) -> tuple[int, int]:
    vocab_size = metadata.vocab_size
    mask_token_id = metadata.mask_token_id
    use_drm = bool(config.arch.__pydantic_extra__.get("discrete_diffusion_init", False))  # type: ignore[union-attr]

    if mask_token_id is None:
        if use_drm:
            mask_token_id = vocab_size
            vocab_size = vocab_size + 1
            if rank == 0:
                print(
                    "[warning] dataset mask_token_id is missing for DRM; "
                    f"assuming legacy ARC metadata and reserving mask token id {mask_token_id} "
                    f"(vocab_size {metadata.vocab_size}->{vocab_size})."
                )
        else:
            mask_token_id = metadata.pad_id
            if rank == 0:
                print(f"[warning] dataset mask_token_id is missing; defaulting to pad_id={mask_token_id}")

    if mask_token_id >= vocab_size:
        old_vocab_size = vocab_size
        vocab_size = mask_token_id + 1
        if rank == 0:
            print(
                f"[warning] Expanding vocab_size from {old_vocab_size} to {vocab_size} "
                f"to include mask_token_id={mask_token_id}."
            )

    return vocab_size, mask_token_id


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    vocab_size, mask_token_id = _resolve_model_vocab_and_mask_token_id(
        config,
        train_metadata,
        rank=rank,
    )

    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        mask_token_id=mask_token_id,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        if hasattr(model, "set_state_noise_scheduler"):
            model.set_state_noise_scheduler(_create_state_noise_scheduler())
        elif hasattr(model, "set_scheduler"):
            model.set_scheduler(_create_state_noise_scheduler())
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def _get_results_experiment_name(config: PretrainConfig) -> str:
    if config.run_name is not None and len(config.run_name.strip()):
        return config.run_name.strip()

    if config.checkpoint_path is not None:
        checkpoint_name = os.path.basename(os.path.normpath(config.checkpoint_path))
        if checkpoint_name:
            return checkpoint_name

    if config.load_checkpoint is not None:
        checkpoint_name = os.path.basename(os.path.normpath(config.load_checkpoint))
        if checkpoint_name:
            return checkpoint_name

    return "unnamed_experiment"


def _get_results_root(config: PretrainConfig) -> str:
    return os.path.join("results", _get_results_experiment_name(config))


def _get_step_results_dir(config: PretrainConfig, step: int) -> str:
    return os.path.join(_get_results_root(config), f"step_{step}")


def _get_step_predictions_dir(config: PretrainConfig, step: int) -> str:
    return os.path.join(_get_step_results_dir(config, step), "predictions")


def _get_step_trajectories_dir(config: PretrainConfig, step: int) -> str:
    return os.path.join(_get_step_results_dir(config, step), "trajectories")


def _get_step_trajectory_spool_dir(config: PretrainConfig, step: int) -> str:
    return os.path.join(_get_step_trajectories_dir(config, step), ".spool")


def _get_step_evaluators_dir(config: PretrainConfig, step: int) -> str:
    return os.path.join(_get_step_results_dir(config, step), "evaluators")


def _normalize_eval_trajectory_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "off": "off",
        "none": "off",
        "no": "off",
        "disabled": "off",
        "selected": "selected",
        "chosen": "selected",
        "voted": "selected",
        "selected by voting": "selected",
        "selected trajectory": "selected",
        "selected trajectories": "selected",
        "all": "all",
        "everything": "all",
        "all examples": "all",
        "save all trajectories": "all",
        "debug": "debug",
    }
    if normalized not in aliases:
        raise ValueError(
            "eval_trajectory_mode must be one of: "
            "'off', 'selected', 'all', or 'debug'."
        )
    return aliases[normalized]


def _init_eval_trajectory_items(batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    batch_size = batch["inputs"].shape[0]
    items: List[Dict[str, Any]] = []
    for batch_idx in range(batch_size):
        item: Dict[str, Any] = {
            "inputs": batch["inputs"][batch_idx].detach().to(torch.uint8).cpu(),
            "labels": batch["labels"][batch_idx].detach().to(torch.int16).cpu(),
            "puzzle_identifier": batch["puzzle_identifiers"][batch_idx].detach().cpu(),
            "predictions": [],
            "q_halt_probs": [],
            "q_halt_logits": [],
            "timesteps": [],
        }
        if "puzzle_indices" in batch:
            item["puzzle_index"] = batch["puzzle_indices"][batch_idx].detach().cpu()
        if "group_indices" in batch:
            item["group_index"] = batch["group_indices"][batch_idx].detach().cpu()
        items.append(item)
    return items


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_compatible(v) for v in value]
    return value


def _record_eval_trajectory_step(
    trajectory_items: List[Dict[str, Any]],
    preds: Dict[str, torch.Tensor],
    inference_step: int,
    timestep: Optional[int] = None,
) -> None:
    q_logits = preds.get("q_halt_logits")
    q_probs = None
    if q_logits is not None:
        q_probs = q_logits.detach().to(torch.float32).sigmoid().cpu()
        q_logits = q_logits.detach().to(torch.float32).cpu()

    for batch_idx, item in enumerate(trajectory_items):
        pred_cpu = preds["preds"][batch_idx].detach().to(torch.uint8).cpu()
        item["predictions"].append(pred_cpu)

        if q_probs is not None and q_logits is not None:
            q_prob_cpu = float(q_probs[batch_idx].item())
            q_logit_cpu = float(q_logits[batch_idx].item())
            item["q_halt_probs"].append(q_prob_cpu)
            item["q_halt_logits"].append(q_logit_cpu)

        item["timesteps"].append(timestep)


def _finalize_eval_trajectory_item(
    item: Dict[str, Any],
    *,
    set_name: str,
    batch_index: int,
    batch_item_index: int,
) -> Dict[str, Any]:
    labels = item["labels"]
    valid_mask = labels != -100

    final_pred = item["predictions"][-1] if len(item["predictions"]) else None
    is_final_correct = False
    is_any_step_correct = False
    correct_steps: List[int] = []
    if valid_mask.any() and final_pred is not None:
        final_pred_for_compare = final_pred.to(labels.dtype)
        is_final_correct = bool(torch.equal(final_pred_for_compare[valid_mask], labels[valid_mask]))
        correct_steps = [
            step_id
            for step_id, pred in enumerate(item["predictions"], start=1)
            if torch.equal(pred.to(labels.dtype)[valid_mask], labels[valid_mask])
        ]
        is_any_step_correct = len(correct_steps) > 0

    item["set_name"] = set_name
    item["batch_index"] = batch_index
    item["batch_item_index"] = batch_item_index
    item["num_inference_steps"] = len(item["predictions"])
    item["is_final_correct"] = is_final_correct
    item["is_any_step_correct"] = is_any_step_correct
    item["correct_steps"] = correct_steps
    item["first_correct_step"] = correct_steps[0] if len(correct_steps) else None
    if len(item["predictions"]):
        item["predictions"] = torch.stack(item["predictions"], dim=0)
    return item


def _save_eval_trajectories(
    config: PretrainConfig,
    train_state: TrainState,
    trajectories: List[Dict[str, Any]],
    *,
    use_drm_eval: bool,
    save_mode: str,
) -> None:
    if config.checkpoint_path is None or not len(trajectories):
        return

    trajectories_dir = _get_step_trajectories_dir(config, train_state.step)
    os.makedirs(trajectories_dir, exist_ok=True)
    payload = {
        "step": train_state.step,
        "mode": "drm" if use_drm_eval else "trm",
        "save_mode": save_mode,
        "trajectories": trajectories,
    }
    with open(os.path.join(trajectories_dir, f"{save_mode}.json"), "w", encoding="utf-8") as handle:
        json.dump(_to_json_compatible(payload), handle)


def _get_base_model(model: nn.Module) -> nn.Module:
    # Unwrap torch.compile and loss head wrappers.
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # type: ignore[attr-defined]
    return getattr(model, "model", model)


def _get_model_state_prefix(model: nn.Module) -> str:
    """
    Infer the wrapper prefix used by the current model state_dict, e.g.
    "_orig_mod.model.", "model.", or "".
    """
    candidate_prefixes = ("_orig_mod.model.", "model.", "_orig_mod.", "")
    state_keys = list(model.state_dict().keys())

    for prefix in candidate_prefixes:
        for key in state_keys:
            if key.startswith(prefix) and key[len(prefix):].startswith("core."):
                return prefix

    return ""


def _upgrade_legacy_checkpoint_keys(state_dict: dict, model: nn.Module) -> dict:
    """
    Translate legacy TRM/SPRM/DRM checkpoint keys to the refactored naming scheme.
    This keeps compatibility localized to checkpoint loading rather than adding
    aliases throughout the refactored model code.
    """
    candidate_prefixes = ("_orig_mod.model.", "model.", "_orig_mod.", "")
    target_prefix = _get_model_state_prefix(model)

    def split_prefix(key: str) -> tuple[str, str]:
        for prefix in candidate_prefixes:
            if key.startswith(prefix):
                return prefix, key[len(prefix):]
        return "", key

    def translate_body(body: str) -> str:
        if body.startswith("inner."):
            body = "core." + body[len("inner."):]

        body = body.replace("L_level.", "recurrent_block.")
        body = body.replace("H_init", "decode_init")
        body = body.replace("L_init", "scratchpad_init")
        body = body.replace("diffusion_scheduler.", "masking_scheduler.")
        return body

    upgraded_state_dict = {}
    migrated_key_count = 0
    saw_legacy_keys = False
    saw_embed_tokens_labels = False
    saw_unsupported_layout = False
    embed_tokens_source_key = None

    for key, value in state_dict.items():
        source_prefix, body = split_prefix(key)
        if body.startswith(("inner.", "L_level.", "H_init", "L_init", "diffusion_scheduler.")):
            saw_legacy_keys = True
        if "L_level_list." in body:
            saw_unsupported_layout = True

        new_body = translate_body(body)
        new_key = f"{target_prefix}{new_body}"
        if new_key != key:
            migrated_key_count += 1

        if new_body == "core.embed_tokens.weight":
            embed_tokens_source_key = new_key
        if new_body == "core.embed_tokens_labels.weight":
            saw_embed_tokens_labels = True

        upgraded_state_dict[new_key] = value

    if saw_unsupported_layout:
        raise ValueError(
            "This checkpoint appears to come from a legacy architecture with untied inner "
            "recurrence blocks (`L_level_list`), which the refactored rm.py does not match. "
            "Use the legacy codepath for that checkpoint or add a dedicated migration."
        )

    if saw_legacy_keys and (not saw_embed_tokens_labels) and (embed_tokens_source_key is not None):
        upgraded_state_dict[f"{target_prefix}core.embed_tokens_labels.weight"] = upgraded_state_dict[
            embed_tokens_source_key
        ].clone()
        print("Copied legacy embed_tokens weights into embed_tokens_labels for compatibility.")

    if migrated_key_count > 0:
        print(f"Remapped {migrated_key_count} checkpoint parameter name(s) to the refactored model.")

    return upgraded_state_dict


def _resize_puzzle_embedding_if_needed(state_dict: dict, model: nn.Module) -> None:
    base_model = _get_base_model(model)
    if not hasattr(base_model, "puzzle_emb"):
        return
    expected_shape: torch.Size = base_model.puzzle_emb.weights.shape  # type: ignore
    for key in list(state_dict.keys()):
        if key.endswith(".puzzle_emb.weights"):
            puzzle_emb = state_dict[key]
            if puzzle_emb.shape != expected_shape:
                print(
                    "Resetting puzzle embedding as shape is different. "
                    f"Found {puzzle_emb.shape}, Expected {expected_shape}"
                )
                state_dict[key] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )


def _reconcile_checkpoint_shapes(state_dict: dict, model: nn.Module) -> dict:
    model_state = model.state_dict()
    reconciled_state = dict(state_dict)

    ignored_unexpected_keys = []
    for key in list(reconciled_state.keys()):
        if key not in model_state:
            if key.endswith(".puzzle_emb_proj.weight") or key.endswith(".puzzle_emb_proj.bias"):
                ignored_unexpected_keys.append(key)
                del reconciled_state[key]

    if ignored_unexpected_keys:
        print(
            "Ignoring legacy checkpoint parameter(s) not used by the refactored model: "
            + ", ".join(sorted(ignored_unexpected_keys))
        )

    vocab_matrix_suffixes = (
        ".core.embed_tokens.embedding_weight",
        ".core.embed_tokens_labels.embedding_weight",
        ".core.lm_head.weight",
    )

    resized_keys = []
    for key, expected_value in model_state.items():
        if key not in reconciled_state:
            continue

        checkpoint_value = reconciled_state[key]
        if checkpoint_value.shape == expected_value.shape:
            continue

        if (
            key.endswith(vocab_matrix_suffixes)
            and checkpoint_value.ndim == 2
            and expected_value.ndim == 2
            and checkpoint_value.shape[1] == expected_value.shape[1]
        ):
            old_vocab_size, hidden_size = checkpoint_value.shape
            new_vocab_size = expected_value.shape[0]
            if old_vocab_size > new_vocab_size:
                reconciled_state[key] = checkpoint_value[:new_vocab_size].contiguous()
                resized_keys.append(f"{key}: truncated vocab rows {old_vocab_size}->{new_vocab_size}")
            else:
                pad_rows = new_vocab_size - old_vocab_size
                pad_value = checkpoint_value.mean(dim=0, keepdim=True).expand(pad_rows, hidden_size)
                reconciled_state[key] = torch.cat([checkpoint_value, pad_value], dim=0).contiguous()
                resized_keys.append(f"{key}: padded vocab rows {old_vocab_size}->{new_vocab_size}")

    if resized_keys:
        print("Adjusted checkpoint tensor shapes for compatibility:")
        for msg in resized_keys:
            print(f"  {msg}")

    return reconciled_state


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        state_dict = _upgrade_legacy_checkpoint_keys(state_dict, model)
        _resize_puzzle_embedding_if_needed(state_dict, model)
        state_dict = _reconcile_checkpoint_shapes(state_dict, model)
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None
    trajectory_mode = _normalize_eval_trajectory_mode(config.eval_trajectory_mode)
    supports_curated_trajectories = any(hasattr(evaluator, "update_trajectory_batch") for evaluator in evaluators)
    save_all_trajectories = trajectory_mode == "all"
    collect_trajectories = save_all_trajectories or (
        trajectory_mode in {"selected", "debug"} and supports_curated_trajectories
    )
    if rank == 0 and save_all_trajectories:
        print("[warning] eval_trajectory_mode=all can use significant disk space.")
    if rank == 0 and trajectory_mode in {"selected", "debug"}:
        if not supports_curated_trajectories:
            print(
                f"[warning] eval_trajectory_mode={trajectory_mode} requires an evaluator "
                "that supports trajectory curation; no curated trajectory file may be written."
            )
    base_model = _get_base_model(train_state.model)
    use_drm_eval = hasattr(base_model, "uses_drm_inference") and base_model.uses_drm_inference()
    drm_timesteps = base_model.get_drm_inference_timesteps() if use_drm_eval else []
    _eval_vocab_size, drm_mask_token_id = _resolve_model_vocab_and_mask_token_id(
        config,
        eval_metadata,
        rank=rank,
    )
    if use_drm_eval and len(drm_timesteps) == 0:
        raise ValueError("DRM inference is enabled but produced zero timesteps.")

    # DRM inference uses random masking/noise during the timestep loop.
    # Scope and seed RNG here so checkpoint-loaded eval is reproducible for a given step.
    fork_devices = [torch.cuda.current_device()] if (use_drm_eval and torch.cuda.is_available()) else []
    with torch.random.fork_rng(devices=fork_devices, enabled=use_drm_eval):
        if use_drm_eval:
            eval_seed = int(config.seed + train_state.step + rank * 100003)
            torch.manual_seed(eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed)

        with torch.inference_mode():
            return_keys = set(config.eval_save_outputs)
            if use_drm_eval:
                # DRM eval updates carry from logits each timestep.
                return_keys.add("logits")
            if collect_trajectories:
                return_keys.update({"preds", "q_halt_logits"})
            for evaluator in evaluators:
                if hasattr(evaluator, "set_trajectory_spool_dir"):
                    spool_dir = None
                    # In curated modes we do not know the voted winner until ARC aggregation
                    # finishes, so stream compact per-candidate trajectory representatives to
                    # disk and load only the chosen examples at the end.
                    if collect_trajectories and trajectory_mode in {"selected", "debug"} and config.checkpoint_path is not None:
                        spool_dir = _get_step_trajectory_spool_dir(config, train_state.step)
                    evaluator.set_trajectory_spool_dir(spool_dir, rank)
                evaluator.begin_eval()
                return_keys.update(evaluator.required_outputs)

            # Run evaluation
            set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

            save_preds = {}
            saved_trajectories: List[Dict[str, Any]] = []

            metric_keys = []
            metric_values = None

            carry = None
            processed_batches = 0
        
            for set_name, batch, global_batch_size in eval_loader:
                processed_batches += 1
                if rank == 0:
                    print(
                        "[eval] "
                        f"processing test batch {processed_batches} | "
                        f"set={set_name} | effective_global_batch={global_batch_size}"
                    )
            
                # To device
                batch = {k: v.cuda() for k, v in batch.items()}
                with torch.device("cuda"):
                    carry = train_state.model.initial_carry(batch)  # type: ignore

                # Forward
                inference_steps = 0
                all_finish = False
                trajectory_items: Optional[List[Dict[str, Any]]] = None
                should_collect_trajectories = collect_trajectories
                if should_collect_trajectories:
                    trajectory_items = _init_eval_trajectory_items(batch)

                if use_drm_eval:
                    drm_puzzle_prefix = base_model.get_drm_eval_puzzle_prefix(carry)
                    for timestep in drm_timesteps:
                        carry, loss, metrics, preds, all_finish = train_state.model(
                            carry=carry, batch=batch, return_keys=return_keys
                        )
                        carry = base_model.update_drm_eval_carry(
                            carry=carry,
                            preds=preds,
                            timestep=timestep,
                            mask_token_id=drm_mask_token_id,
                            puzzle_prefix=drm_puzzle_prefix,
                        )
                        inference_steps += 1
                        if trajectory_items is not None:
                            _record_eval_trajectory_step(
                                trajectory_items,
                                preds,
                                inference_step=inference_steps,
                                timestep=int(timestep),
                            )
                else:
                    while True:
                        carry, loss, metrics, preds, all_finish = train_state.model(
                            carry=carry, batch=batch, return_keys=return_keys
                        )
                        inference_steps += 1
                        if trajectory_items is not None:
                            _record_eval_trajectory_step(
                                trajectory_items,
                                preds,
                                inference_step=inference_steps,
                            )

                        if all_finish:
                            break

                if rank == 0:
                    print(f"[eval]   completed model inference in {inference_steps} step(s)")

                for collection in (batch, preds):
                    for k, v in collection.items():
                        if k in config.eval_save_outputs:
                            save_preds.setdefault(k, [])
                            save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

                for evaluator in evaluators:
                    evaluator.update_batch(batch, preds)

                if trajectory_items is not None:
                    finalized_trajectories = []
                    for batch_item_index, item in enumerate(trajectory_items):
                        finalized = _finalize_eval_trajectory_item(
                            item,
                            set_name=set_name,
                            batch_index=processed_batches,
                            batch_item_index=batch_item_index,
                        )
                        finalized_trajectories.append(finalized)

                    for evaluator in evaluators:
                        if hasattr(evaluator, "update_trajectory_batch"):
                            evaluator.update_trajectory_batch(finalized_trajectories)

                    if save_all_trajectories:
                        saved_trajectories.extend(finalized_trajectories)
                    del finalized_trajectories
                    del trajectory_items

                del carry, loss, preds, batch, all_finish

                # Aggregate metrics
                set_id = set_ids[set_name]

                if metric_values is None:
                    metric_keys = list(
                        sorted(metrics.keys())
                    )  # Sort keys to guarantee all processes use the same order.
                    metric_values = torch.zeros(
                        (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                    )

                metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

                del metrics

            # concatenate save preds
            save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

            # Save preds
            if config.checkpoint_path is not None and len(save_preds):
                if world_size > 1:
                    gathered_save_preds = [None for _ in range(world_size)] if rank == 0 else None
                    dist.gather_object(save_preds, gathered_save_preds, dst=0)
                else:
                    gathered_save_preds = [save_preds]

                if rank == 0:
                    combined_save_preds: Dict[str, List[torch.Tensor]] = {}
                    for rank_preds in gathered_save_preds:  # type: ignore[union-attr]
                        if rank_preds is None:
                            continue
                        for key, value in rank_preds.items():
                            combined_save_preds.setdefault(key, [])
                            combined_save_preds[key].append(value)

                    merged_save_preds = {
                        key: torch.cat(values, dim=0) for key, values in combined_save_preds.items()
                    }

                    predictions_dir = _get_step_predictions_dir(config, train_state.step)
                    os.makedirs(predictions_dir, exist_ok=True)
                    torch.save(
                        merged_save_preds,
                        os.path.join(predictions_dir, "all_preds.pt"),
                    )

            del save_preds

            if save_all_trajectories:
                if world_size > 1:
                    gathered_trajectories = [None for _ in range(world_size)] if rank == 0 else None
                    dist.gather_object(saved_trajectories, gathered_trajectories, dst=0)
                else:
                    gathered_trajectories = [saved_trajectories]

                if rank == 0:
                    merged_trajectories: List[Dict[str, Any]] = []
                    for rank_trajectories in gathered_trajectories:  # type: ignore[union-attr]
                        if rank_trajectories is not None:
                            merged_trajectories.extend(rank_trajectories)

                    _save_eval_trajectories(
                        config,
                        train_state,
                        merged_trajectories,
                        use_drm_eval=use_drm_eval,
                        save_mode="all",
                    )

            # Reduce to rank 0
            if metric_values is not None:
                if world_size > 1:
                    dist.reduce(metric_values, dst=0)

                if rank == 0:
                    reduced_metrics = metric_values.cpu().numpy()
                    reduced_metrics = {
                        set_name: {
                            metric_name: reduced_metrics[set_id, metric_id]
                            for metric_id, metric_name in enumerate(metric_keys)
                        }
                        for set_id, set_name in enumerate(set_ids)
                    }

                    # Postprocess
                    for set_name, m in reduced_metrics.items():
                        count = m.pop("count")
                        reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

            # Run evaluators
            if rank == 0:
                print(f"\n[eval] Running {len(evaluators)} evaluator(s) on aggregated outputs...")
            
            for i, evaluator in enumerate(evaluators):
                if rank == 0:
                    print(f"[eval] Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
                # Path for saving
                evaluator_save_path = None
                if config.checkpoint_path is not None:
                    evaluators_dir = _get_step_evaluators_dir(config, train_state.step)
                    evaluator_save_path = os.path.join(
                        evaluators_dir,
                        evaluator.__class__.__name__,
                    )
                    os.makedirs(evaluator_save_path, exist_ok=True)

                # Run and log
                metrics = evaluator.result(
                    evaluator_save_path,
                    rank=rank,
                    world_size=world_size,
                    group=cpu_group,
                    trajectory_mode=trajectory_mode,
                )
                if rank == 0 and metrics is not None:
                    if reduced_metrics is None:
                        reduced_metrics = {}

                    reduced_metrics.update(metrics)
                    print(f"[eval]   completed {evaluator.__class__.__name__}")
                    
            if rank == 0:
                print("[eval] All evaluators completed.")

            # Curated trajectory saving uses a shared on-disk spool directory.
            # Nonzero ranks return early from evaluator.result(), so keep all ranks
            # synchronized here before any rank can start the next eval/training
            # phase and reuse or delete the same spool path.
            if (
                world_size > 1
                and collect_trajectories
                and trajectory_mode in {"selected", "debug"}
                and config.checkpoint_path is not None
            ):
                dist.barrier(group=cpu_group)

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="configs", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print(
                "[train] Starting streamed training batches "
                f"for epoch chunk {_iter_id + 1}/{total_iters}; "
                f"current optimizer step={train_state.step}/{train_state.total_steps}."
            )
        train_state.model.train()
        train_batches_processed = 0
        for set_name, batch, global_batch_size in train_loader:
            train_batches_processed += 1
            if RANK == 0 and (train_batches_processed == 1 or train_batches_processed % 100 == 0):
                print(
                    "[train] "
                    f"processing batch {train_batches_processed} in this epoch chunk | "
                    f"set={set_name} | effective_global_batch={global_batch_size} | "
                    f"completed_optimizer_steps={train_state.step}/{train_state.total_steps}"
                )
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print(
                    "[eval] Starting full evaluation sweep over test batches "
                    f"for set(s): {_format_set_list(eval_metadata.sets)}."
                )
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
