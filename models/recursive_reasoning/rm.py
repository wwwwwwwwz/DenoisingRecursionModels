from typing import Tuple, List, Dict
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding
from diffusion.schedulers import DiscreteScheduler

IGNORE_LABEL_ID = -100

@dataclass
class RecurrentState:
    """Per-step recurrence state (decode_latent is decoded; scratchpad_latent is not)."""
    decode_latent: torch.Tensor
    scratchpad_latent: torch.Tensor


@dataclass
class ACTLoopState:
    """ACT wrapper state (tracks halting + active batch data)."""
    recurrent_state: RecurrentState

    step_count: torch.Tensor
    is_halted: torch.Tensor

    active_batch: Dict[str, torch.Tensor]


class RecursiveReasoningConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    mask_token_id: int

    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True
    diffusion_num_training_steps: int = 1000
    diffusion_num_inference_steps: int = 20
    diffusion_schedule: str = 'linear'
    diffusion_noise_type: str = 'uniform'
    diffusion_inference_confidence_masking: bool = False

    state_perturbation: bool = False
    # DRM-style: reinitialize recurrence state from diffusion/inference every step.
    discrete_diffusion_init: bool = False
    noise_scaling: str = 'latents'

    @property
    def warmup_cycles(self) -> int:
        """Number of warm-up recurrences executed under no_grad."""
        return max(self.H_cycles - 1, 0)

    @property
    def grad_cycles(self) -> int:
        """Number of recurrences executed inside the gradient window."""
        return self.L_cycles

class Layer(nn.Module):
    def __init__(self, config: RecursiveReasoningConfig) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            if self.config.puzzle_emb_len == 0:
                self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            else:
                self.puzzle_emb_len = self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class RecurrentBlock(nn.Module):
    """Stack of layers that is applied repeatedly during recurrence."""
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class RecursiveReasoningCore(nn.Module):
    """Core recurrence model (embeddings, heads, and recurrent block)."""
    def __init__(self, config: RecursiveReasoningConfig) -> None:
        super().__init__()
        self.config = config
        self.masking_scheduler = DiscreteScheduler(
            num_train_timesteps=self.config.diffusion_num_training_steps,
            num_inference_timesteps=self.config.diffusion_num_inference_steps,
            schedule=config.diffusion_schedule,
            noise_type=config.diffusion_noise_type,
            confidence_masking=self.config.diffusion_inference_confidence_masking
        )

        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.embed_tokens_labels = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        if self.config.puzzle_emb_len == 0:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        else:
            self.puzzle_emb_len = self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.recurrent_block = RecurrentBlock(
            layers=[Layer(self.config) for _i in range(self.config.L_layers)]
        )

        self.decode_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.scratchpad_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _embed_inputs(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input)

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )
        return self.embed_scale * embedding

    def _embed_labels(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens_labels(input)

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )
        return self.embed_scale * embedding

    def _time_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            timesteps: (B,) tensor of time indices.
            dim: embedding dimension.

        Returns:
            (B, dim) tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def empty_state(self, batch_size: int):
        return RecurrentState(
            decode_latent=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            scratchpad_latent=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_state(self, reset_flag: torch.Tensor, carry: RecurrentState):
        return RecurrentState(
            decode_latent=torch.where(
                reset_flag.view(-1, 1, 1), self.decode_init, carry.decode_latent
            ),
            scratchpad_latent=torch.where(
                reset_flag.view(-1, 1, 1), self.scratchpad_init, carry.scratchpad_latent
            ),
        )

    def diffusion_state(self, batch):
        batch_size = batch["labels"].shape[0]
        device = batch["labels"].device
        safe_labels = batch["labels"].clone()

        safe_labels[safe_labels == IGNORE_LABEL_ID] = 0
        noise = self.masking_scheduler.get_noise(safe_labels, device)

        timesteps = torch.randint(
            0, self.masking_scheduler.num_train_timesteps, (batch_size,), device=device
        ).long()
        mask_ratios = self.masking_scheduler.get_mask_ratios(timesteps).unsqueeze(-1)
        mask = noise < mask_ratios

        safe_labels = torch.where(mask, self.config.mask_token_id, safe_labels)

        total_embedding = self._embed_labels(safe_labels, batch["puzzle_identifiers"])

        return RecurrentState(
            decode_latent=total_embedding,
            scratchpad_latent=(
                self.scratchpad_init
                .to(device)
                .expand(batch_size, total_embedding.shape[1], -1)
                .clone()
            ),
        )

    def inference_state(self, batch):
        batch_size = batch["labels"].shape[0]
        device = batch["labels"].device

        masked_tensor = torch.full_like(
            input=batch["labels"],
            fill_value=self.config.mask_token_id,
            dtype=torch.long,
        )

        final_embed = self._embed_labels(masked_tensor, batch["puzzle_identifiers"])

        return RecurrentState(
            decode_latent=final_embed,
            scratchpad_latent=(
                self.scratchpad_init
                .to(device)
                .expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1)
                .clone()
            ),
        )

    @torch.no_grad
    def noise_latent(self, latent, scheduler):
        """Add scaled noise to a latent using the provided noise scheduler."""
        batch_size = latent.shape[0]

        if self.config.noise_scaling == 'token_embeds':
            embed_w = self.embed_tokens.embedding_weight
            mean = embed_w.mean(dim=0, keepdim=True)
            std = embed_w.std(dim=0, keepdim=True).clamp_min(1e-6)
        elif self.config.noise_scaling == 'latents':
            mean = latent.mean(dim=(0, 1), keepdim=True)
            std = latent.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

        eps = torch.randn_like(latent)
        eps.mul_(std).add_(mean)
        timesteps = torch.randint(
            low=0, high=int(scheduler.config.num_train_timesteps * 0.1),
            size=(batch_size,), device=latent.device, dtype=torch.long,
        )

        return scheduler.add_noise(latent, eps, timesteps)

    def forward(
        self,
        carry: RecurrentState,
        batch: Dict[str, torch.Tensor],
        state_noise_scheduler=None,
    ) -> Tuple[RecurrentState, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._embed_inputs(batch["inputs"], batch["puzzle_identifiers"])
        decode_latent, scratchpad_latent = carry.decode_latent, carry.scratchpad_latent

        # Warm-up recurrences (no grad). These advance the recursion state before the gradient window.
        with torch.no_grad():
            for _warmup_step in range(self.config.warmup_cycles):
                for _grad_step in range(self.config.grad_cycles):
                    scratchpad_latent = self.recurrent_block(
                        scratchpad_latent, decode_latent + input_embeddings, **seq_info
                    )
                decode_latent = self.recurrent_block(decode_latent, scratchpad_latent, **seq_info)

        # Gradient window recurrences (with grad).
        for _grad_step in range(self.config.grad_cycles):
            scratchpad_latent = self.recurrent_block(
                scratchpad_latent, decode_latent + input_embeddings, **seq_info
            )
        decode_latent = self.recurrent_block(decode_latent, scratchpad_latent, **seq_info)

        # Only decode from the answer latent.
        output = self.lm_head(decode_latent)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(decode_latent[:, 0]).to(torch.float32)

        # Apply a single perturbation at the end of the window (SPRM-style),
        # after decoding.
        if not torch.is_inference_mode_enabled() and self.config.state_perturbation:
            decode_latent = self.noise_latent(decode_latent, state_noise_scheduler)
            scratchpad_latent = self.noise_latent(scratchpad_latent, state_noise_scheduler)

        new_state = RecurrentState(
            decode_latent=decode_latent.detach(),
            scratchpad_latent=scratchpad_latent.detach(),
        )
        return new_state, output, (q_logits[..., 0], q_logits[..., 1])

class RecursiveReasoningModel(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = RecursiveReasoningConfig(**config_dict)
        self.core = RecursiveReasoningCore(self.config)
        self.state_noise_scheduler = None

    @property
    def puzzle_emb(self):
        return self.core.puzzle_emb

    def set_state_noise_scheduler(self, scheduler):
        """Set the scheduler used for state perturbation noise."""
        self.state_noise_scheduler = scheduler

    def uses_drm_inference(self) -> bool:
        """Whether evaluation should run the DRM timestep loop."""
        return self.config.discrete_diffusion_init

    def get_drm_inference_timesteps(self) -> List[int]:
        """Return DRM timesteps for iterative evaluation denoising."""
        return self.core.masking_scheduler.get_timesteps()

    def get_drm_eval_puzzle_prefix(self, carry: ACTLoopState) -> torch.Tensor:
        """Freeze the puzzle prefix once per eval batch (pretrain_drm behavior)."""
        return carry.recurrent_state.decode_latent[:, :self.core.puzzle_emb_len, :].clone()

    @torch.no_grad()
    def update_drm_eval_carry(
        self,
        carry: ACTLoopState,
        preds: Dict[str, torch.Tensor],
        timestep: int,
        mask_token_id: int,
        puzzle_prefix: torch.Tensor,
    ) -> ACTLoopState:
        """
        Apply one DRM remasking step to predictions and inject back into decode_latent.
        """
        scheduler = self.core.masking_scheduler
        grid_predictions = preds["preds"].clone()
        device = grid_predictions.device

        if scheduler.confidence_masking:
            logits = preds["logits"]
            non_zero = grid_predictions != 0
            num_non_zero = non_zero.sum(dim=1)

            proportion = float(timestep) / scheduler.num_train_timesteps
            num_to_mask = (proportion * num_non_zero.float()).int()
            num_to_mask = torch.clamp(
                num_to_mask,
                min=torch.tensor(0, device=device),
                max=num_non_zero,
            )

            for batch_idx in range(grid_predictions.shape[0]):
                total_to_mask = num_to_mask[batch_idx].item()
                if total_to_mask == 0:
                    continue

                non_zero_positions = torch.where(non_zero[batch_idx])[0]
                logits_non_zero = logits[batch_idx][non_zero[batch_idx]]
                confidences = torch.softmax(logits_non_zero, dim=-1).max(dim=-1).values

                num_low_confidence = int(0.99 * total_to_mask)
                num_random = total_to_mask - num_low_confidence

                sorted_confidence_indices = torch.argsort(confidences)
                low_conf_indices = sorted_confidence_indices[:num_low_confidence]

                remaining_indices = sorted_confidence_indices[num_low_confidence:]
                random_perm = torch.randperm(remaining_indices.shape[0], device=device)
                random_indices = remaining_indices[random_perm[:num_random]]

                combined_indices = torch.cat([low_conf_indices, random_indices])
                positions_to_mask = non_zero_positions[combined_indices]
                grid_predictions[batch_idx, positions_to_mask] = mask_token_id
        else:
            noise = scheduler.get_noise(grid_predictions, device=device)
            mask = noise < float(timestep) / scheduler.num_train_timesteps
            grid_predictions = torch.where(mask, mask_token_id, grid_predictions)

        previous_y_embedding = self.core.embed_tokens_labels(grid_predictions)
        decode_latent = torch.cat([puzzle_prefix, previous_y_embedding], dim=1)
        new_state = RecurrentState(
            decode_latent=decode_latent,
            scratchpad_latent=carry.recurrent_state.scratchpad_latent,
        )
        return ACTLoopState(
            recurrent_state=new_state,
            step_count=carry.step_count,
            is_halted=carry.is_halted,
            active_batch=carry.active_batch,
        )

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        if self.training:
            recurrent_state = self.core.diffusion_state(batch)
        else:
            recurrent_state = self.core.inference_state(batch)

        return ACTLoopState(
            recurrent_state=recurrent_state,
            step_count=torch.zeros((batch_size, ), dtype=torch.int32),
            is_halted=torch.zeros((batch_size, ), dtype=torch.bool),
            active_batch={k: v.clone() for k, v in batch.items()}
        )

    def _init_recurrent_state_for_batch(self, batch: Dict[str, torch.Tensor]) -> RecurrentState:
        """Initialize recurrence state for a fresh batch (train vs. eval behavior)."""
        if self.training:
            return self.core.diffusion_state(batch)
        return self.core.inference_state(batch)

    def _merge_active_batch(self, carry: ACTLoopState, new_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        pretrain.py streams new batches every step. For any item that has not halted,
        keep its previous data so the recursion continues on the same example.
        """
        halted = carry.is_halted
        return {
            k: torch.where(
                halted.view((-1, ) + (1, ) * (new_batch[k].ndim - 1)),
                new_batch[k],
                v
            )
            for k, v in carry.active_batch.items()
        }

    def forward(self, carry: ACTLoopState, batch: Dict[str, torch.Tensor]) -> Tuple[ACTLoopState, Dict[str, torch.Tensor]]:

        if self.config.discrete_diffusion_init and self.training:
            # DRM training: each call gets a freshly diffused state.
            new_recurrent_state = self._init_recurrent_state_for_batch(batch)
            new_steps = torch.zeros_like(carry.step_count)
            new_active_batch = batch
        elif self.config.discrete_diffusion_init:
            # DRM eval: keep recurrent_state so update_drm_eval_carry can drive iterative denoising.
            new_recurrent_state = carry.recurrent_state
            new_steps = carry.step_count
            new_active_batch = batch
        else:
            new_recurrent_state = carry.recurrent_state
            new_steps = torch.where(carry.is_halted, 0, carry.step_count)
            new_active_batch = self._merge_active_batch(carry, batch)


        new_recurrent_state, logits, (q_halt_logits, q_continue_logits) = self.core(
            new_recurrent_state,
            new_active_batch,
            state_noise_scheduler=self.state_noise_scheduler
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():

            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.config.discrete_diffusion_init:
                if self.training:
                    halted = torch.ones_like(carry.is_halted, dtype=torch.bool)
                else:
                    halted = is_last_step
            elif self.training and (self.config.halt_max_steps > 1):

                halt_by_q_only = self.config.no_ACT_continue
                if halt_by_q_only:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return ACTLoopState(new_recurrent_state, new_steps, halted, new_active_batch), outputs
