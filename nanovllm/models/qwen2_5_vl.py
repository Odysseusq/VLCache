from typing import Tuple
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from einops import rearrange
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from nanovllm.models.qwen3 import Qwen3Model
from flash_attn import flash_attn_varlen_func


def permute_inv(perm: torch.Tensor) -> torch.Tensor:
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv_perm


class VisionQKVLinear(QKVParallelLinear):
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str = None):
        if loaded_shard_id is None:
            # Assume loaded_weight is the full packed QKV
            param_data = param.data
            
            # Calculate total sizes
            total_num_heads = self.num_heads * self.tp_size
            total_num_kv_heads = self.num_kv_heads * self.tp_size
            head_dim = self.head_size
            
            # Loaded weight layout: [Q, K, V]
            q_size = total_num_heads * head_dim
            k_size = total_num_kv_heads * head_dim
            v_size = total_num_kv_heads * head_dim
            
            if loaded_weight.shape[0] != (q_size + k_size + v_size):
                if loaded_weight.ndim == 1:
                    # Bias loading
                    q = loaded_weight.narrow(0, 0, q_size)
                    k = loaded_weight.narrow(0, q_size, k_size)
                    v = loaded_weight.narrow(0, q_size + k_size, v_size)
                    
                    q_shard_size = self.num_heads * head_dim
                    q_shard = q.narrow(0, self.tp_rank * q_shard_size, q_shard_size)
                    
                    k_shard_size = self.num_kv_heads * head_dim
                    k_shard = k.narrow(0, self.tp_rank * k_shard_size, k_shard_size)
                    
                    v_shard_size = self.num_kv_heads * head_dim
                    v_shard = v.narrow(0, self.tp_rank * v_shard_size, v_shard_size)
                    
                    shard = torch.cat([q_shard, k_shard, v_shard], dim=0)
                    param_data.copy_(shard)
                    return

            # Weight loading
            q = loaded_weight.narrow(0, 0, q_size)
            k = loaded_weight.narrow(0, q_size, k_size)
            v = loaded_weight.narrow(0, q_size + k_size, v_size)
            
            # Get shards
            q_shard_size = self.num_heads * head_dim
            q_shard = q.narrow(0, self.tp_rank * q_shard_size, q_shard_size)
            
            k_shard_size = self.num_kv_heads * head_dim
            k_shard = k.narrow(0, self.tp_rank * k_shard_size, k_shard_size)
            
            v_shard_size = self.num_kv_heads * head_dim
            v_shard = v.narrow(0, self.tp_rank * v_shard_size, v_shard_size)
            
            # Concatenate shards [Q_shard, K_shard, V_shard]
            shard = torch.cat([q_shard, k_shard, v_shard], dim=0)
            
            param_data.copy_(shard)
        else:
            super().weight_loader(param, loaded_weight, loaded_shard_id)


class RotaryPosMixin:
    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        if isinstance(h, torch.Tensor):
            h = int(h.item())
        if isinstance(w, torch.Tensor):
            w = int(w.item())
        if isinstance(spatial_merge_size, torch.Tensor):
            spatial_merge_size = int(spatial_merge_size.item())
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))


class VisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        proj_bias: bool = True,
        flatten_batch: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.projection_size = projection_size
        self.flatten_batch = flatten_batch
        self.use_qkv_parallel = use_qkv_parallel
        
        tp_size = dist.get_world_size()
        self.num_heads_per_partition = num_heads // tp_size
        self.q_size = self.num_heads_per_partition * self.head_dim
        self.kv_size = self.num_heads_per_partition * self.head_dim

        if use_qkv_parallel:
            self.qkv = VisionQKVLinear(
                hidden_size=embed_dim,
                head_size=self.head_dim,
                total_num_heads=num_heads,
                total_num_kv_heads=num_heads,
                bias=True,
            )
        else:
            raise NotImplementedError("Only QKV parallel linear is supported for now")
            
        self.proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=proj_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        output_ws=None,
    ) -> torch.Tensor:
        # x: [B, S, H]
        bsz, s, _ = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q = q.reshape(bsz * s, self.num_heads_per_partition, self.head_dim).contiguous()
        k = k.reshape(bsz * s, self.num_heads_per_partition, self.head_dim).contiguous()
        v = v.reshape(bsz * s, self.num_heads_per_partition, self.head_dim).contiguous()

        cos, sin = position_embeddings
        
        def apply_rotary_pos_emb_vision(t, cos, sin):
            # t: [N, H, D]
            # cos, sin: [N, D]
            t_rot = (t * cos.unsqueeze(1)) + (rotate_half(t) * sin.unsqueeze(1))
            return t_rot

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q = apply_rotary_pos_emb_vision(q, cos, sin)
        k = apply_rotary_pos_emb_vision(k, cos, sin)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False
        )
        
        output = output.reshape(bsz, s, -1)
        output = self.proj(output)
        
        return output


class Qwen2_5_VLMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
        )
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act(gate_up)
        x_down = self.down_proj(x)
        return x_down


class Qwen2_5_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        S, B, H = x.shape
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s h -> s b h")

        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        
        mlp_out = self.mlp(x_norm_2d)
        x = x_after_add_2d + mlp_out
        
        return x.reshape(S, B, H)


class Qwen2_5_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: [S, B, context_dim]
        S, B, D = x.shape
        x2d = x.reshape(-1, D)
        x2d = self.ln_q(x2d)  # RMSNorm expects 2D
        x2d = x2d.view(-1, self.hidden_size)  # group into spatial_merge_unit
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel = mlp_fc1(x2d)
        x_parallel = mlp_act(x_parallel)
        out = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionTransformer(nn.Module, RotaryPosMixin):
    def __init__(
        self,
        vision_config: Qwen2_5_VLConfig, # using Qwen2_5_VLVisionConfig actually
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        mlp_hidden_size: int = vision_config.intermediate_size
        self.out_hidden_size = vision_config.out_hidden_size
        
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_eps=norm_eps,
                )
                for i in range(depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def get_window_index(self, grid_thw):
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        window_index: list = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb_fn(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        rotary_pos_emb = self.rot_pos_emb_fn(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # Move window_index to the same device as x before using it to index x
        window_index = window_index.to(device=x.device)
        reverse_indices = permute_inv(window_index)

        # Ensure rotary_pos_emb is on the same device/dtype as x
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        position_embeddings = (
            position_embeddings[0].to(x.device, x.dtype),
            position_embeddings[1].to(x.device, x.dtype),
        )

        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
                .cumsum(dim=0)
                .to(device=x.device, dtype=torch.int32),
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        x = x.unsqueeze(1) # [S, 1, D] - batch dim is 1 here
        
        for layer_num, blk in enumerate(self.blocks):
            fullatt_indexes = self.fullatt_block_indexes
            if isinstance(fullatt_indexes, torch.Tensor):
                fullatt_indexes = fullatt_indexes.tolist()
            if layer_num in fullatt_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # adapter
        x = self.merger(x)
        x = x[reverse_indices, :]

        return x


from nanovllm.layers.embed_head import ParallelLMHead

class Qwen2_5_VLForConditionalGeneration(nn.Module):
    # Mapping for loading weights
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
    ) -> None:
        super().__init__()
        self.config = config
        
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def get_visual_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Get input embeddings from LLM
        inputs_embeds = self.model.embed_tokens(input_ids)
        
        if pixel_values is not None:
            visual_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # Fill in visual embeddings
            image_token_id = getattr(self.config, "image_token_id", 151655)
            image_mask = (input_ids == image_token_id)
            
            # Verify shapes matches
            assert image_mask.sum() == visual_embeds.shape[0], f"Shape mismatch: {image_mask.sum()} vs {visual_embeds}"
            inputs_embeds[image_mask] = visual_embeds.to(inputs_embeds.dtype)
            
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
