from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from diffusion.schedulers import DiscreteScheduler
import os
import numpy as np
import json

import sys

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
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
    no_ACT_continue: bool =  True


    diffusion_num_training_steps: int = 1000
    diffusion_num_inference_steps: int = 20
    diffusion_schedule: str = 'linear'
    diffusion_noise_type: str = 'uniform'
    diffusion_inference_confidence_masking: bool = False

    noise_latent_H_step: bool = False
    noise_latent_L_step: bool = False
    noise_scaling: str = 'latents'

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
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
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:

            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)

        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.diffusion_scheduler = DiscreteScheduler(num_train_timesteps=self.config.diffusion_num_training_steps, num_inference_timesteps=self.config.diffusion_num_inference_steps,schedule=config.diffusion_schedule, noise_type=config.diffusion_noise_type, confidence_masking=self.config.diffusion_inference_confidence_masking)

        self.forward_dtype = getattr(torch, self.config.forward_dtype)


        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.embed_tokens_labels = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:

            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)


        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass


        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])


        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)


        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input)


        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)


        if self.config.pos_encodings == "learned":

            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))


        return self.embed_scale * embedding

    def _label_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens_labels(input)


        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)


        if self.config.pos_encodings == "learned":

            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))


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

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def diffusion_carry(self, batch):
        batch_size = batch["labels"].shape[0]
        device = batch["labels"].device
        safe_labels = batch["labels"].clone()

        safe_labels[safe_labels== IGNORE_LABEL_ID] = 0


        noise = self.diffusion_scheduler.get_noise(safe_labels, device)

        timesteps = torch.randint(0, self.diffusion_scheduler.num_train_timesteps, (batch_size,), device=device).long()
        mask_ratios = self.diffusion_scheduler.get_mask_ratios(timesteps).unsqueeze(-1)
        mask = noise < mask_ratios

        safe_labels = torch.where(mask, self.config.mask_token_id, safe_labels)

        total_embedding = self._label_embeddings(safe_labels, batch["puzzle_identifiers"])

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=total_embedding,
            z_L=self.L_init.to(device).expand(batch_size, total_embedding.shape[1], -1).clone(),
        )


    def inference_carry(self, batch):
        batch_size = batch["labels"].shape[0]
        device = batch["labels"].device

        masked_tensor = torch.full_like(
            input=batch['labels'],
            fill_value=self.config.mask_token_id,
            dtype=torch.long
        )

        final_embed = self._label_embeddings(masked_tensor, batch["puzzle_identifiers"])

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=final_embed,
            z_L=self.L_init.to(device).expand(batch_size, self.config.seq_len+self.puzzle_emb_len, -1).clone(),
        )

    @torch.no_grad
    def noise_latent(self, latent, scheduler):
        batch_size = latent.shape[0]

        if self.config.noise_scaling == 'token_embeds':
            embed_w = self.embed_tokens.embedding_weight
            mean = embed_w.mean(dim=0, keepdim=True)
            std  = embed_w.std(dim=0, keepdim=True).clamp_min(1e-6)
        elif self.config.noise_scaling == 'latents':
            mean = latent.mean(dim=(0, 1), keepdim=True)
            std  = latent.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

        eps = torch.randn_like(latent)
        eps.mul_(std).add_(mean)


        timesteps = torch.randint(
            low=0, high=int(scheduler.config.num_train_timesteps * 0.1),
            size=(batch_size,), device=latent.device, dtype=torch.long
        )

        return scheduler.add_noise(latent, eps, timesteps)


    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], noise_scheduler) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])


        it = 0
        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    if not torch.is_inference_mode_enabled() and self.config.noise_latent_L_step:
                        z_L = self.noise_latent(z_L, noise_scheduler)
                z_H = self.L_level(z_H, z_L, **seq_info)
                if not torch.is_inference_mode_enabled() and self.config.noise_latent_H_step:
                    z_H = self.noise_latent(z_H, noise_scheduler)
                    z_L = self.noise_latent(z_L, noise_scheduler)

        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)


        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)
        self.noise_scheduler = None

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def set_scheduler(self, scheduler):
        self.noise_scheduler = scheduler

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        if self.training:
            inner_carry = self.inner.diffusion_carry(batch)
        else:
            inner_carry = self.inner.inference_carry(batch)

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=inner_carry,
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.zeros((batch_size, ), dtype=torch.bool),
            current_data={k: v.clone() for k, v in batch.items()}
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:


        new_inner_carry = carry.inner_carry

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}


        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data,noise_scheduler=self.noise_scheduler)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():

            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step


            if self.training and (self.config.halt_max_steps > 1):


                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)


                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:


                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
