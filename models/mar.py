from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from timm.models.vision_transformer import Block, Mlp, DropPath
from typing import List, Optional, Sequence, Tuple, Iterable
from models.diffloss import DiffLoss
from util.misc import is_main_process
from models.confidence import VarianceConfidenceAccumulator, standardize, load_confidence_stats
from models.confidence.variance_confidence import (
                TrajectoryConfidenceAccumulator, 
                VarianceConfidenceAccumulator,
                CosineTrajectoryAccumulator,
                SemanticConsistencyAccumulator
            )

# =========================================================================
# RoPE 核心函数
# =========================================================================

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """为广播操作重塑频率张量的形状"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.ndim < 3:
        # 适配 [B, L, H, D] 格式，在 H (head) 维度广播
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        shape = [d if i == 1 or i == ndim - 1 or i == 0 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用旋转位置编码 (复数乘法实现)
    Args:
        xq: [B, L, H, D]
        xk: [B, L, H, D]
        freqs_cis: [L, D] (complex64)
    """
    # 将输入转换为复数形式: (..., D) -> (..., D/2, 2) -> complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 广播 freqs_cis
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # 复数乘法应用旋转，然后转回实数并展平
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Confidence helpers
def _parse_conf_window(conf_window: Optional[str], num_timesteps: int) -> List[int]:
    """
    Parse confidence window specification to a list of timesteps.
    Supports formats like "40:10" (start=40, step=10 descending) or comma-separated integers.
    """
    if conf_window is None or conf_window == "":
        return []

    if isinstance(conf_window, (list, tuple, set)):
        items = list(conf_window)
    else:
        conf_window = str(conf_window)
        if "," in conf_window:
            items = conf_window.split(",")
        elif ":" in conf_window:
            parts = conf_window.split(":")
            nums = [int(p) for p in parts if p != ""]
            if len(nums) == 2:
                start, step = nums
                step = abs(step) if step != 0 else 1
                items = list(range(start, -1, -step))
            elif len(nums) == 3:
                start, end, step = nums
                step = abs(step) if step != 0 else 1
                if start <= end:
                    items = list(range(start, min(end, num_timesteps - 1) + 1, step))
                else:
                    items = list(range(start, max(end, 0) - 1, -step))
            else:
                items = nums
        else:
            items = [conf_window]

    steps: List[int] = []
    for item in items:
        try:
            step_int = int(item)
        except ValueError:
            continue
        if 0 <= step_int < num_timesteps:
            steps.append(step_int)

    return sorted(list(set(steps)), reverse=False)

# 自定义支持 RoPE 的 Transformer Block
class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, return_entropy=False):
        B, N, C = x.shape
        # qkv: [3, B, num_heads, N, head_dim]
        # 注意这里 permute 成了 (2, 0, 3, 1, 4) -> [3, B, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # === 注入 RoPE ===
        if freqs_cis is not None:
            # apply_rotary_emb 期望输入为 [B, L, H, D]
            # 当前 q, k 是 [B, H, L, D]，需要 permute
            q = q.transpose(1, 2) # -> [B, L, H, D]
            k = k.transpose(1, 2)
            q, k = apply_rotary_emb(q, k, freqs_cis)
            q = q.transpose(1, 2) # -> [B, H, L, D]
            k = k.transpose(1, 2)
        # ============================
        
        entropy = None
        if return_entropy:
            # 1. 手动计算 Attention 以获取分布 (为了求熵)
            # q, k: [B, H, N, D]
            # 使用 float32 保证数值稳定性
            attn_logits = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            attn_probs = attn_logits.softmax(dim=-1)
            
            # 2. 计算香农熵: -Sum(p * log(p))
            eps = 1e-10
            # dim=-1 是对每个 token 的关注分布求熵
            entropy_per_head = -torch.sum(attn_probs * torch.log(attn_probs + eps), dim=-1)
            
            # 3. 对 Head 维度求平均，得到每个 Token 的平均熵 [B, N]
            entropy = entropy_per_head.mean(dim=1)
            
            # 4. 计算输出 (恢复原有精度)
            x = attn_probs.to(q.dtype) @ v
            if self.training:
                x = self.attn_drop(x)
        else:
            # 保持原有的高效实现
            dropout_p = self.attn_drop.p if self.training else 0.0
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, scale=self.scale
            )
        # 调整输出形状 (SDPA 输出是 [B, H, N, D])
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, entropy
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def project_kv(self, cond_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = cond_tokens.shape
        kv = self.kv(cond_tokens).reshape(bsz, seq_len, 2, self.num_heads, dim // self.num_heads)
        k, v = kv.permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x: torch.Tensor, cond_tokens: Optional[torch.Tensor] = None,
                freqs_cis_q: Optional[torch.Tensor] = None,
                freqs_cis_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q(x).reshape(bsz, seq_len, self.num_heads, dim // self.num_heads).transpose(1, 2)

        if cond_tokens is None:
            raise ValueError("cond_tokens must be provided for cross attention.")
        k, v = self.project_kv(cond_tokens)

        if freqs_cis_q is not None:
            q = q.transpose(1, 2)
            q, _ = apply_rotary_emb(q, q, freqs_cis_q)
            q = q.transpose(1, 2)

        if freqs_cis_k is not None:
            k = k.transpose(1, 2)
            k, _ = apply_rotary_emb(k, k, freqs_cis_k)
            k = k.transpose(1, 2)

        dropout_p = self.attn_drop.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            scale=self.scale
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, dim)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out

class RoPEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_lr_inject=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # 使用自定义的 RoPEAttention
        self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 2. MLP (Feed-Forward)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.enable_lr_inject = use_lr_inject
        if use_lr_inject:
            self.cross_attn = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )
            self.norm_cross = norm_layer(dim)
            self.lr_inject_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attn = None
            self.norm_cross = None
            self.lr_inject_gate = None
            
    def forward(self, x, freqs_cis=None, cond_tokens=None, gate_multiplier=1.0,
                cross_attn_freqs_q=None, cross_attn_freqs_k=None, return_entropy=False):
        # 将 freqs_cis 传入 Attention
        attn_out, entropy = self.attn(self.norm1(x), freqs_cis=freqs_cis, return_entropy=return_entropy)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.cross_attn is not None and cond_tokens is not None:
            if not torch.is_tensor(gate_multiplier):
                gate_multiplier = x.new_tensor(gate_multiplier)
            cross_out = self.cross_attn(
                self.norm_cross(x),
                cond_tokens=cond_tokens,
                freqs_cis_q=cross_attn_freqs_q,
                freqs_cis_k=cross_attn_freqs_k
            )
            x = x + self.drop_path(cross_out) * gate_multiplier * self.lr_inject_gate
        return x, entropy
    
def precompute_freqs_cis_2d(dim: int, coords_h, coords_w, theta: float = 10000.0):
    """
    根据给定的 H 和 W 坐标预计算 2D RoPE 频率
    """
    # dim 分给 X 和 Y 各一半 (dim//4 * 2 = dim//2 in complex)
    # 计算基础频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)).to(coords_h.device)
    
    # 生成外积：坐标 * 频率
    freqs_h = torch.outer(coords_h, freqs) # [L, dim//4]
    freqs_w = torch.outer(coords_w, freqs) # [L, dim//4]
    
    # 转为极坐标复数形式 (Modulus=1, Angle=freqs)
    freqs_cis_h = torch.polar(torch.ones_like(freqs_h), freqs_h) # complex64
    freqs_cis_w = torch.polar(torch.ones_like(freqs_w), freqs_w) # complex64
    
    # 拼接 H 和 W 的编码 -> [L, dim//2] (complex)
    freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=1) 
    return freqs_cis

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    pos: [N] tensor
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (N,)
    out = torch.einsum('m,d->md', pos, omega)  # (N, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (N, D)
    return emb

def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    """
    grid: [2, H, W]
    """
    assert embed_dim % 2 == 0
    # 前一半维度编码 H (y轴)，后一半维度编码 W (x轴)
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0].flatten())
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1].flatten())
    
    emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed_torch(embed_dim, grid_size_hr, grid_size_lr, device, base_size=16):
    """
    生成 2D 绝对位置编码，支持任意分辨率归一化。
    
    Args:
        embed_dim: 输出维度
        grid_size_hr: tuple (h_hr, w_hr) -> HR 特征图的 (高, 宽)
        grid_size_lr: tuple (h_lr, w_lr) -> LR 特征图的 (高, 宽)
        device: 设备
        base_size: 归一化基准大小 (建议设为训练时 token 的默认边长，如 16)
    """
    h_hr, w_hr = grid_size_hr
    h_lr, w_lr = grid_size_lr
    
    # === 归一化坐标系 ===
    # 逻辑：(坐标 + 0.5) / 当前边长 * 基准边长
    # 效果：无论分辨率由多大，都被映射到 [0, base_size] 的连续空间
    
    # 1. 生成 HR 网格 (Pixel Center Alignment)
    grid_h = (torch.arange(h_hr, dtype=torch.float, device=device) + 0.5) / h_hr * base_size
    grid_w = (torch.arange(w_hr, dtype=torch.float, device=device) + 0.5) / w_hr * base_size
    grid_x, grid_y = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid_hr = torch.stack([grid_y, grid_x], dim=0) # [2, H, W]

    # 2. 生成 LR 网格
    # LR 网格在同一空间下的坐标。
    # 只要 HR 和 LR 覆盖相同的物理视野（FOV），这种归一化就能保证它们空间对齐。
    grid_l_h = (torch.arange(h_lr, dtype=torch.float, device=device) + 0.5) / h_lr * base_size
    grid_l_w = (torch.arange(w_lr, dtype=torch.float, device=device) + 0.5) / w_lr * base_size
    grid_lx, grid_ly = torch.meshgrid(grid_l_w, grid_l_h, indexing='xy')
    grid_lr = torch.stack([grid_ly, grid_lx], dim=0) # [2, h_lr, w_lr]

    # 3. 计算 Sin/Cos 编码 (调用原有函数)
    pos_embed_hr = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid_hr)
    pos_embed_lr = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid_lr)

    # 4. 拼接 (Buffer/LR 在前)
    pos_embed = torch.cat([pos_embed_lr, pos_embed_hr], dim=0).unsqueeze(0)
    return pos_embed

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,   #这里需要设置为LR转码后的长度
                 diffloss_d=6,      #显存紧张可以降低
                 diffloss_w=1024,   #可以与embed_dim保持一致
                 num_sampling_steps='100',
                 diffusion_batch_mul=1,         #单卡最好设为1，之前为4
                 grad_checkpointing=False,
                 mse_weight=0.2,
                 use_lr_inject=False,
                 lr_inject_layers="all",
                 lr_inject_cond_source="encoder",
                 use_rope=False,
                 use_mse_loss=False,
                 use_dynamic_maskgit: bool = False,
                 conf_threshold: float = 0.0,
                 conf_pmin: float = 0.01,
                 conf_window: str = "40:10",
                 conf_method='entropy',
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        self.mse_weight = mse_weight
        self.use_lr_inject = use_lr_inject
        self.lr_inject_cond_source = lr_inject_cond_source
        self.use_rope = use_rope
        self.use_mse_loss = use_mse_loss
        self.use_dynamic_maskgit = use_dynamic_maskgit
        self.conf_threshold = conf_threshold
        self.conf_pmin = conf_pmin
        self.conf_window = conf_window
        self.conf_method = conf_method
        self.remask_mode = "mask_token"
        self.conf_stats_path = None
        self._conf_stats_cache = None
        self._confidence_window_cache = None
        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics

        # 投影层：把 VAE 的 16 维向量映射到 Transformer 的 1024 维
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        #投影层：将 Decoder 维度 (768) 映射到 Token 维度 (16)
        self.final_proj = nn.Linear(decoder_embed_dim, self.token_embed_dim)
        # 缓冲区大小：定义前缀长度 (64)
        self.buffer_size = buffer_size
        # 位置编码：这是一个可学习的参数表，长度 = 序列长度(256) + 缓冲区长度(64) = 320，维度 = 1024
        if not self.use_rope:
            lr_token_len = self.seq_h * self.seq_w
            total_tokens = lr_token_len + self.seq_len
            self.encoder_pos_embed_learned = nn.Parameter(
                torch.zeros(1, total_tokens, encoder_embed_dim)
            )
        # Transformer Blocks：堆叠 16 层
        # self.encoder_blocks = nn.ModuleList([
        #     Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
        #           proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        encoder_inject_layers = self._resolve_lr_inject_layers(
            encoder_depth, lr_inject_layers
        )
        self.encoder_blocks = nn.ModuleList([
            RoPEBlock(
                encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop=proj_dropout, attn_drop=attn_dropout,
                use_lr_inject=self.use_lr_inject and i in encoder_inject_layers
            ) for i in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)
        # --------------------------------------------------------------------------
        # MAR decoder specifics

        # 投影层：如果 Encoder 和 Decoder 维度不一样，这里负责转换 (通常是一样的)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        # mask_token：这是一个特殊的向量，代表“未知”，所有被遮住的位置，都会填入这个向量
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Decoder 位置编码
        if not self.use_rope:
            lr_token_len = self.seq_h * self.seq_w
            total_tokens = lr_token_len + self.seq_len
            self.decoder_pos_embed_learned = nn.Parameter(
                torch.zeros(1, total_tokens, decoder_embed_dim)
            )

        # Transformer Blocks：堆叠 16 层
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
        #           proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        decoder_inject_layers = self._resolve_lr_inject_layers(
            decoder_depth, lr_inject_layers
        )
        self.decoder_blocks = nn.ModuleList([
             RoPEBlock(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop=proj_dropout, attn_drop=attn_dropout,
                use_lr_inject=self.use_lr_inject and i in decoder_inject_layers
            ) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Diffusion 位置编码：这是给 DiffLoss 用的额外位置信息
        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))
        self.initialize_weights()
        # --------------------------------------------------------------------------
        # Diffusion Loss
        # 实例化 DiffLoss 模块，它是一个独立的子网络 (MLP 或 Transformer)
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        if self.use_lr_inject:
            source_dim = self._infer_lr_inject_source_dim(encoder_embed_dim)
            self.lr_inject_cond_proj_encoder = self._build_cond_proj(
                source_dim, encoder_embed_dim
            )
            self.lr_inject_cond_proj_decoder = self._build_cond_proj(
                source_dim, decoder_embed_dim
            )
            if not self.use_rope:
                lr_seq_len = self.seq_h * self.seq_w
                self.lr_pos_embed_encoder = nn.Parameter(
                    torch.zeros(1, lr_seq_len, encoder_embed_dim)
                )
                self.lr_pos_embed_decoder = nn.Parameter(
                    torch.zeros(1, lr_seq_len, decoder_embed_dim)
                )
        self._lr_inject_rope_logged = False

    def _maybe_log_lr_inject_rope(self, num_img_tokens: int, shape_hr: Tuple[int, int]) -> None:
        if not self.use_lr_inject or self._lr_inject_rope_logged:
            return
        if not is_main_process():
            return
        h_tokens, w_tokens = shape_hr
        skip_special_tokens = False
        print(
            f"[LR Inject RoPE] N_img={num_img_tokens} Ht={h_tokens} Wt={w_tokens} "
            f"skip_special_tokens={skip_special_tokens}"
        )
        self._lr_inject_rope_logged = True
    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        self.apply(self._init_weights)
        #初始化投影层
        torch.nn.init.xavier_uniform_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            torch.nn.init.constant_(self.final_proj.bias, 0)


    def _infer_lr_inject_source_dim(self, encoder_embed_dim):
        if self.lr_inject_cond_source == "encoder":
            return encoder_embed_dim
        if self.lr_inject_cond_source == "patch_embed":
            return self.token_embed_dim
        if self.lr_inject_cond_source == "vae_latent":
            return self.vae_embed_dim
        raise ValueError(f"Unsupported lr_inject_cond_source: {self.lr_inject_cond_source}")

    @staticmethod
    def _build_cond_proj(source_dim: int, target_dim: int) -> nn.Module:
        if source_dim == target_dim:
            return nn.Identity()
        return nn.Linear(source_dim, target_dim)

    def _resolve_lr_inject_layers(self, depth: int, mode: str) -> Sequence[int]:
        if mode == "all":
            return set(range(depth))
        if mode == "first_half":
            return set(range(depth // 2))
        if mode == "last_half":
            start = depth - depth // 2
            return set(range(start, depth))
        raise ValueError(f"Unsupported lr_inject_layers mode: {mode}")

    def _build_lr_inject_cond_tokens(self, x_lr, lr_tokens=None):
        if self.lr_inject_cond_source == "vae_latent":
            cond_tokens = x_lr.permute(0, 2, 3, 1).reshape(x_lr.shape[0], -1, x_lr.shape[1])
        else:
            if lr_tokens is None:
                lr_tokens = self.patchify(x_lr)
            if self.lr_inject_cond_source == "patch_embed":
                cond_tokens = lr_tokens
            elif self.lr_inject_cond_source == "encoder":
                cond_tokens = self.z_proj_ln(self.z_proj(lr_tokens))
            else:
                raise ValueError(f"Unsupported lr_inject_cond_source: {self.lr_inject_cond_source}")
        return cond_tokens

    def get_lr_inject_gate_mean(self) -> float:
        if not self.use_lr_inject:
            return 0.0
        gates = []
        for block in list(self.encoder_blocks) + list(self.decoder_blocks):
            gate = getattr(block, "lr_inject_gate", None)
            if gate is not None:
                gates.append(gate.detach().mean())
        if not gates:
            return 0.0
        return torch.stack(gates).mean().item()
    
    def _get_confidence_window(self) -> List[int]:
        if hasattr(self, "_confidence_window_cache") and self._confidence_window_cache is not None:
            return self._confidence_window_cache
        num_steps = self.diffloss.gen_diffusion.num_timesteps
        self._confidence_window_cache = _parse_conf_window(self.conf_window, num_steps)
        return self._confidence_window_cache

    def _get_confidence_stats(self, device: torch.device):
        """
        Modified to support Trajectory Confidence (returns sigma_delta).
        """
        if self._conf_stats_cache is None:
            # 修改为你校准文件的实际路径
            fname = "pretrained_models/40_10_traj/confidence_stats.npz"
            stats = load_confidence_stats(fname)
            
            if stats is None:
                # Fallback
                self._conf_stats_cache = None
            else:
                # 检查返回值类型
                if isinstance(stats, tuple):
                    # 旧版 (mu, sigma)
                    print("Loaded Variance Confidence Stats (Legacy).")
                    self._conf_stats_cache = (stats[0].float(), stats[1].float())
                else:
                    # 新版 (sigma_delta)
                    print("Loaded Trajectory Confidence Stats (Sigma Delta).")
                    self._conf_stats_cache = stats.float() # [1, C, 1, 1]

        stats = self._conf_stats_cache
        if stats is None:
            return None
            
        # Move to device
        if isinstance(stats, tuple):
            return stats[0].to(device), stats[1].to(device)
        else:
            return stats.to(device)
    
    def _init_weights(self, m):
        # 初始化全连接层和归一化层的bias和weight
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # 切片，图片->序列
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]   输出形状: [Batch, Seq_Len=256, Dim=16]

    def unpatchify(self, x, shape=None):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        
        # 优先使用传入的动态形状
        if shape is not None:
            h_, w_ = shape
        else:
            # 只有在没传 shape 时，才尝试去猜（回退到训练时的固定形状）
            h_, w_ = self.seq_h, self.seq_w

        # (可选) 校验一下是否匹配
        # assert h_ * w_ == x.shape[1], f"Shape mismatch: tokens={x.shape[1]}, target={h_}x{w_}"

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz, num_tokens):
        # generate a batch of random generation orders
        # 如果没传，回退到默认长度
        if num_tokens is None:
            num_tokens = self.seq_len

        orders = []
        for _ in range(bsz):
            order = np.array(list(range(num_tokens)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        # 执行遮挡 (把指定位置设为 1/Masked)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask
    
    # [新增] 生成 2D RoPE 频率的方法
    def get_rope_freqs(self, shape_hr, shape_lr, device, head_dim):
        """
        生成 LR + HR 拼接后的 2D RoPE 频率
        输出形状: [1, L_total, D//2] (complex)
        """
        h_hr, w_hr = shape_hr
        h_lr, w_lr = shape_lr
        
        # 1. 生成归一化坐标 (类似 VARSR 使用 meshgrid)
        # 使用 self.seq_h 作为 base_size 进行归一化，保持尺度一致性
        base_size = self.seq_h
        
        # HR 坐标
        grid_h_hr = (torch.arange(h_hr, device=device) + 0.5) / h_hr * base_size
        grid_w_hr = (torch.arange(w_hr, device=device) + 0.5) / w_hr * base_size
        mesh_y_hr, mesh_x_hr = torch.meshgrid(grid_h_hr, grid_w_hr, indexing='ij')
        coords_h_hr = mesh_y_hr.flatten()
        coords_w_hr = mesh_x_hr.flatten()

        # LR 坐标
        grid_h_lr = (torch.arange(h_lr, device=device) + 0.5) / h_lr * base_size
        grid_w_lr = (torch.arange(w_lr, device=device) + 0.5) / w_lr * base_size
        mesh_y_lr, mesh_x_lr = torch.meshgrid(grid_h_lr, grid_w_lr, indexing='ij')
        coords_h_lr = mesh_y_lr.flatten()
        coords_w_lr = mesh_x_lr.flatten()

        # 拼接: LR (buffer) 在前，然后是 HR
        coords_h = torch.cat([coords_h_lr, coords_h_hr], dim=0)
        coords_w = torch.cat([coords_w_lr, coords_w_hr], dim=0)

        # 2. 计算频率 (调用第一步定义的函数)
        freqs_cis = precompute_freqs_cis_2d(head_dim, coords_h, coords_w)
        
        # 增加 Batch 维度 [L, D//2]
        return freqs_cis
    
    def forward_mae_encoder(self, x_hr, mask, x_lr, shape_hr, shape_lr,
                            cond_tokens=None, gate_multiplier=1.0):
        # x: [Batch, Seq_Len=256, Dim=16] (VAE 输出的 token)
        # mask: [Batch, Seq_Len=256] (0=可见, 1=遮挡)
        # x_lr: LR tokens（来自VAE编码，维度 16）
        hr_embedding = self.z_proj(x_hr)
        lr_embedding = self.z_proj(x_lr)
        bsz, seq_len, embed_dim = hr_embedding.shape

        # 获取 head_dim
        num_heads = self.encoder_blocks[0].attn.num_heads
        head_dim = embed_dim // num_heads

        # 拼接 Features
        x = torch.cat([lr_embedding, hr_embedding], dim=1)

        num_lr_tokens = shape_lr[0] * shape_lr[1]
        if self.use_rope:
            freqs_cis_full = self.get_rope_freqs(shape_hr, shape_lr, x.device, head_dim)
            freqs_cis_full = freqs_cis_full.unsqueeze(0).repeat(bsz, 1, 1) # 扩展到 Batch
            freqs_cis_lr = freqs_cis_full[:, :num_lr_tokens]
        else:
            freqs_cis_full = None
            freqs_cis_lr = None

        # 构造 Mask (LR 永远可见)
        mask_with_buffer = torch.cat([torch.zeros(bsz, num_lr_tokens, device=x.device), mask], dim=1)

        x = self.z_proj_ln(x)
        if not self.use_rope:
            pos_embed = self.encoder_pos_embed_learned[:, : x.shape[1], :]
            x = x + pos_embed
        # Dropping (同时筛选 Feature 和 频率)
        keep_indices = (1 - mask_with_buffer).nonzero(as_tuple=True)
        
        # 计算保留的 token 数量
        num_kept = int(mask_with_buffer.shape[1] - mask_with_buffer.sum(dim=1)[0].item())
        
        # 筛选 Feature
        x = x[keep_indices].reshape(bsz, num_kept, embed_dim)
        
        # 筛选 Frequencies
        freqs_cis_kept = None if freqs_cis_full is None else freqs_cis_full[keep_indices].reshape(bsz, num_kept, -1)

        # Apply Transformer Blocks (传入 freqs_cis)
        if cond_tokens is not None:
            assert cond_tokens.shape[0] == x.shape[0], "cond_tokens batch mismatch with encoder tokens."
            assert cond_tokens.shape[-1] == embed_dim, "cond_tokens dim mismatch with encoder embed dim."
        
        for idx, block in enumerate(self.encoder_blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():         
                gate_multiplier_tensor = x.new_tensor(gate_multiplier)
                # [关键修改] 这里必须要解包，因为 block 返回的是 (x, entropy)
                x, _ = checkpoint(
                    block, x, freqs_cis_kept, cond_tokens, gate_multiplier_tensor,
                    freqs_cis_kept, freqs_cis_lr
                )
            else:
                # [关键修改] 这里必须要解包
                x, _ = block(x, freqs_cis=freqs_cis_kept, cond_tokens=cond_tokens,
                          gate_multiplier=gate_multiplier,
                          cross_attn_freqs_q=freqs_cis_kept, cross_attn_freqs_k=freqs_cis_lr)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask, shape_hr, shape_lr,
                            cond_tokens=None, gate_multiplier=1.0, return_entropy=False):
        x = self.decoder_embed(x)
        num_lr_tokens = shape_lr[0] * shape_lr[1]
        bsz = x.shape[0]
        embed_dim = x.shape[-1]
        # 获取 head_dim
        num_heads = self.decoder_blocks[0].attn.num_heads
        head_dim = embed_dim // num_heads

        # Pad Mask Tokens
        mask_with_buffer = torch.cat([torch.zeros(bsz, num_lr_tokens, device=x.device), mask], dim=1)
        mask_tokens = self.mask_token.repeat(bsz, mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        
        keep_indices = (1 - mask_with_buffer).nonzero(as_tuple=True)
        x_after_pad[keep_indices] = x.reshape(-1, embed_dim)
        x = x_after_pad

        if self.use_rope:
            freqs_cis_full = self.get_rope_freqs(shape_hr, shape_lr, x.device, head_dim)
            freqs_cis_full = freqs_cis_full.repeat(bsz, 1, 1) 
        else:
            freqs_cis_full = None
            if not hasattr(self, "decoder_pos_embed_learned"):
                raise RuntimeError("decoder_pos_embed_learned is not initialized when use_rope=False")
            pos_embed = self.decoder_pos_embed_learned[:, : x.shape[1], :]
            x = x + pos_embed

        # Apply Transformer Blocks
        last_layer_idx = len(self.decoder_blocks) - 1
        final_entropy = None

        if cond_tokens is not None:
            assert cond_tokens.shape[0] == x.shape[0], "cond_tokens batch mismatch with decoder tokens."
            assert cond_tokens.shape[-1] == embed_dim, "cond_tokens dim mismatch with decoder embed dim."
            
        for idx, block in enumerate(self.decoder_blocks):
            # 只有最后一层计算 Entropy
            should_return_entropy = return_entropy and (idx == last_layer_idx)
            
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # Grad checkpointing doesn't support multiple outputs easily, assume inference
                gate_multiplier_tensor = x.new_tensor(gate_multiplier)
                # [关键修改] 解包 checkpoint 返回的元组
                ret = checkpoint(
                    block, x, freqs_cis_full, cond_tokens, gate_multiplier_tensor,
                    freqs_cis_full, freqs_cis_full[:, :num_lr_tokens] if freqs_cis_full is not None else None
                )
                if isinstance(ret, tuple):
                    x = ret[0]
                else:
                    x = ret
                
                block_entropy = None # Checkpointing disables this feature
            else:
                x, block_entropy = block(
                    x, freqs_cis=freqs_cis_full, cond_tokens=cond_tokens,
                    gate_multiplier=gate_multiplier,
                    cross_attn_freqs_q=freqs_cis_full,
                    cross_attn_freqs_k=freqs_cis_full[:, :num_lr_tokens] if freqs_cis_full is not None else None,
                    return_entropy=should_return_entropy
                )
            
            if should_return_entropy:
                final_entropy = block_entropy

        x = self.decoder_norm(x)

        # 移除 Buffer (LR tokens)
        x = x[:, num_lr_tokens:]
        if final_entropy is not None:
             final_entropy = final_entropy[:, num_lr_tokens:]

        pos_embed = get_2d_sincos_pos_embed_torch(
            self.decoder_embed.out_features, shape_hr, shape_lr, x.device, base_size=self.seq_h
        )
        pos_embed_hr_only = pos_embed[:, num_lr_tokens:, :]
        pos_embed_hr_only = pos_embed_hr_only.expand(bsz, -1, -1)
        
        return x, pos_embed_hr_only, final_entropy

    def forward_loss(self, z, pos_embed, target, mask):
        # z: Decoder 预测出的 Latent 特征 [Batch, Seq_Len, Dim]
        # target: 真实的 Latent 特征 (Ground Truth) [Batch, Seq_Len, Dim]
        # mask: 当前的遮挡掩码 [Batch, Seq_Len]  
        if self.use_mse_loss:
            z_projected = self.final_proj(z)
            loss_mse_element = (z_projected - target) ** 2
            loss_mse_token = loss_mse_element.mean(dim=-1)
            loss_mse = (loss_mse_token * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss_mse = torch.zeros(1, device=z.device, dtype=z.dtype)
        z_for_diff = z + pos_embed
        bsz, seq_len, _ = target.shape

        # 每一张图片在一次 Forward 中同时学习 4 个不同的扩散时间步（diffusion_batch_mul）
        target_diff = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        
        # 使用加了位置编码的 z_for_diff
        z_diff = z_for_diff.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask_diff = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)

        loss_diff = self.diffloss(z=z_diff, target=target_diff, mask=mask_diff)
        loss = loss_diff + (self.mse_weight * loss_mse if self.use_mse_loss else loss_mse)
        
        return loss, loss_diff, loss_mse

    def forward(self, x_hr, x_lr, gate_multiplier=1.0):
        shape_hr = (self.seq_h, self.seq_w)
        shape_lr = (self.seq_h, self.seq_w)

        # patchify and mask (drop) tokens
        lr_tokens = self.patchify(x_lr)
        hr_tokens = self.patchify(x_hr)
        # 获取当前 batch 的真实 token 数量
        num_tokens = hr_tokens.shape[1]
        self._maybe_log_lr_inject_rope(num_tokens, shape_hr)
        # 备份groundtruth（gt_latents）
        gt_latents = hr_tokens.clone().detach()
        # 生成随机掩码
        orders = self.sample_orders(bsz=hr_tokens.size(0), num_tokens = num_tokens)
        mask = self.random_masking(hr_tokens, orders)
        cond_tokens_encoder = None
        cond_tokens_decoder = None
        if self.use_lr_inject:
            cond_tokens_base = self._build_lr_inject_cond_tokens(x_lr, lr_tokens=lr_tokens)
            cond_tokens_encoder = self.lr_inject_cond_proj_encoder(cond_tokens_base)
            cond_tokens_decoder = self.lr_inject_cond_proj_decoder(cond_tokens_base)
            if not self.use_rope:
                lr_pos_enc = self.lr_pos_embed_encoder[:, : cond_tokens_encoder.shape[1], :]
                lr_pos_dec = self.lr_pos_embed_decoder[:, : cond_tokens_decoder.shape[1], :]
                cond_tokens_encoder = cond_tokens_encoder + lr_pos_enc
                cond_tokens_decoder = cond_tokens_decoder + lr_pos_dec

        # mae encoder
        x = self.forward_mae_encoder(
            hr_tokens, mask, lr_tokens, shape_hr, shape_lr,
            cond_tokens=cond_tokens_encoder,
            gate_multiplier=gate_multiplier
        )
        # mae decoder
        z, pos_embed  = self.forward_mae_decoder(
            x, mask, shape_hr, shape_lr,
            cond_tokens=cond_tokens_decoder,
            gate_multiplier=gate_multiplier
        )

        # lossW
        loss, loss_diff, loss_mse = self.forward_loss(z=z, pos_embed=pos_embed, target=gt_latents, mask=mask)
        
        return loss, loss_diff, loss_mse
        
    def sample_tokens(self, bsz, num_iter=64, x_lr=None,
                      temperature=1.0, progress=False,
                      gate_multiplier=1.0):
        # 必须有 LR 输入
        if x_lr is None:
            raise ValueError("Super-Resolution requires LR input!")
        
        shape_lr = (self.seq_h, self.seq_w)
        shape_hr = (self.seq_h, self.seq_w)
        num_hr_tokens = self.seq_len
        self._maybe_log_lr_inject_rope(num_hr_tokens, shape_hr)
        
        # 先展平 LR tokens
        lr_tokens = self.patchify(x_lr)
        cond_tokens_encoder = None
        cond_tokens_decoder = None
        if self.use_lr_inject:
            cond_tokens_base = self._build_lr_inject_cond_tokens(x_lr, lr_tokens=lr_tokens)
            cond_tokens_encoder = self.lr_inject_cond_proj_encoder(cond_tokens_base)
            cond_tokens_decoder = self.lr_inject_cond_proj_decoder(cond_tokens_base)
            if not self.use_rope:
                lr_pos_enc = self.lr_pos_embed_encoder[:, : cond_tokens_encoder.shape[1], :]
                lr_pos_dec = self.lr_pos_embed_decoder[:, : cond_tokens_decoder.shape[1], :]
                cond_tokens_encoder = cond_tokens_encoder + lr_pos_enc
                cond_tokens_decoder = cond_tokens_decoder + lr_pos_dec

        # init and sample generation orders
        device = x_lr.device 
        mask = torch.ones(bsz, num_hr_tokens, device=device)
        tokens = torch.zeros(bsz, num_hr_tokens, self.token_embed_dim, device=device)
        
        # 初始化实际步数
        actual_steps = num_iter
        
        if not self.use_dynamic_maskgit:
            # === 旧逻辑：固定步数的 MaskGIT (保持不变) ===
            orders = self.sample_orders(bsz, num_tokens=num_hr_tokens)
            indices = list(range(num_iter))
            if progress:
                indices = tqdm(indices)
            prev_mask_len = None
            for step in indices:
                cur_tokens = tokens.clone()
                x = self.forward_mae_encoder(
                    tokens, mask, lr_tokens, shape_hr, shape_lr,
                    cond_tokens=cond_tokens_encoder,
                    gate_multiplier=gate_multiplier
                )
                z, pos_embed = self.forward_mae_decoder(
                    x, mask, shape_hr, shape_lr,
                    cond_tokens=cond_tokens_decoder,
                    gate_multiplier=gate_multiplier
                )
                mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
                mask_len = torch.Tensor([np.floor(num_hr_tokens * mask_ratio)]).cuda()
                mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                         torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
                if prev_mask_len is not None:
                    assert mask_len[0] <= prev_mask_len + 1e-6
                prev_mask_len = mask_len[0]
                mask_next = mask_by_order(mask_len[0], orders, bsz, num_hr_tokens)
                if step >= num_iter - 1:
                    mask_to_pred = mask[:bsz].bool()
                    mask_next = torch.zeros_like(mask)
                else:
                    mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
                mask = mask_next

                indices_to_pred = mask_to_pred.nonzero(as_tuple=True)
                z_sub = z[indices_to_pred]
                pos_sub = pos_embed[indices_to_pred]
                z_cond = z_sub + pos_sub
                sampled_token_latent = self.diffloss.sample(z_cond, temperature)
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
                tokens = cur_tokens.clone()
        else:
            # =======================================================
            # 优化版 Dynamic MaskGIT
            # =======================================================
            

            use_entropy = (self.conf_method == "entropy")
            use_cosine = (self.conf_method == "cosine") 
            use_semantic = (self.conf_method == "semantic")
            # --- 模式判断逻辑 ---
            window_steps = None
            is_trajectory_mode = False
            sigma_delta = None
            mu_u, sigma_u = None, None
            
            if not use_entropy and not use_cosine:
                # Stats Mode (Legacy Variance or Trajectory Energy)
                window_steps = self._get_confidence_window()
                if not window_steps:
                    window_steps = list(range(self.diffloss.gen_diffusion.num_timesteps))
                
                stats = self._get_confidence_stats(tokens.device)
                if stats is not None:
                    if isinstance(stats, tuple):
                        mu_u, sigma_u = stats
                    else:
                        sigma_delta = stats
                        is_trajectory_mode = True
                else:
                    # 默认回退到 Trajectory Energy
                    is_trajectory_mode = True
                    sigma_delta = torch.ones(1, self.token_embed_dim, 1, 1, device=tokens.device)

            if is_main_process():
                if use_entropy:
                    print(f"[Dynamic-MaskGIT] Running in Attention Entropy Mode.")
                elif use_cosine:
                    print(f"[Dynamic-MaskGIT] Running in Cosine Trajectory Mode.")
                elif use_semantic:
                    # 打印日志
                    print(f"[Dynamic-MaskGIT] Running in Feature Semantic Consistency Mode (SR vs LR).")
                elif is_trajectory_mode:
                    print(f"[Dynamic-MaskGIT] Running in Trajectory Stability Mode.")
                else:
                    print(f"[Dynamic-MaskGIT] Running in Variance Mode.")

            round_idx = 0
            unfinished_batch_mask = torch.ones(bsz, dtype=torch.bool, device=tokens.device)
            saved_confidence = torch.full((bsz, num_hr_tokens), -1.0, device=device)
            
            # Remask 参数
            remask_rate = 0.02 # 每次回退 2%
            num_remask = max(1, int(num_hr_tokens * remask_rate))

            # --- 动态阈值配置 ---
            if use_entropy:
                # Attention Entropy: 越小越好 (Range: 0~6)
                start_threshold = max(0.01, self.conf_threshold - 0.2)
                end_threshold = self.conf_threshold + 0.3
                ramp_steps = 6.0
                stochastic_scale = 1.0
                # ===  Cosine Mode 配置 ===
                # Metric 是 (1 - CosineSim). Range [0, 2]
                # 0=一致(好), 1=正交, 2=相反
            elif use_cosine or use_semantic:
                start_threshold = max(0.0, self.conf_threshold - 0.1)
                end_threshold = self.conf_threshold + 0.1
                ramp_steps = 10.0
                stochastic_scale = 0.0
            elif is_trajectory_mode:
                # Energy: 越小越好 (Range: 0+)
                start_threshold = max(0.0, self.conf_threshold - 0.5)
                end_threshold = self.conf_threshold + 0.1
                ramp_steps = 6.0
                stochastic_scale = 0.2
            else:
                # Z-Score: 越小越好 (Range: -3~3)
                start_threshold = self.conf_threshold - 1.0
                end_threshold = self.conf_threshold + 1.5
                ramp_steps = 6.0
                stochastic_scale = 1.0

            while unfinished_batch_mask.any():
                active_indices = torch.nonzero(unfinished_batch_mask).squeeze(1)
                curr_bsz_active = active_indices.shape[0]

                tokens_active = tokens[active_indices]
                mask_active = mask[active_indices]
                lr_tokens_active = lr_tokens[active_indices]
                
                cond_enc_active = cond_tokens_encoder[active_indices] if cond_tokens_encoder is not None else None
                cond_dec_active = cond_tokens_decoder[active_indices] if cond_tokens_decoder is not None else None
                
                gate_mul_active = gate_multiplier
                if torch.is_tensor(gate_multiplier) and gate_multiplier.shape[0] == bsz:
                    gate_mul_active = gate_multiplier[active_indices]

                # --- Forward ---
                z_list = []
                pos_embed_list = []
                entropy_list = []
                
                for i in range(curr_bsz_active):
                    t_i = tokens_active[i:i+1]; m_i = mask_active[i:i+1]; l_i = lr_tokens_active[i:i+1]
                    c_enc_i = cond_enc_active[i:i+1] if cond_enc_active is not None else None
                    c_dec_i = cond_dec_active[i:i+1] if cond_dec_active is not None else None
                    g_mul_i = gate_mul_active[i:i+1] if torch.is_tensor(gate_mul_active) else gate_mul_active

                    x_i = self.forward_mae_encoder(t_i, m_i, l_i, shape_hr, shape_lr, cond_tokens=c_enc_i, gate_multiplier=g_mul_i)
                    z_i, pos_i, ent_i = self.forward_mae_decoder(x_i, m_i, shape_hr, shape_lr, cond_tokens=c_dec_i, gate_multiplier=g_mul_i, return_entropy=use_entropy)
                    z_list.append(z_i); pos_embed_list.append(pos_i)
                    if use_entropy: entropy_list.append(ent_i)
                
                z_active = torch.cat(z_list, dim=0)
                pos_embed_active = torch.cat(pos_embed_list, dim=0)
                entropy_active = torch.cat(entropy_list, dim=0) if use_entropy else None

                mask_to_pred_active = mask_active.bool()
                indices_to_pred_active = mask_to_pred_active.nonzero(as_tuple=True)
                
                if indices_to_pred_active[0].numel() == 0:
                    unfinished_batch_mask[active_indices] = False
                    continue

                z_sub_active = z_active[indices_to_pred_active]
                pos_sub_active = pos_embed_active[indices_to_pred_active]
                z_cond_active = z_sub_active + pos_sub_active

                # --- 采样与置信度获取 ---
                u_map = None
                sampled_token_latent = None
                
                if use_entropy:
                    # Entropy Mode
                    sampled_token_latent = self.diffloss.sample(
                        z_cond_active, temperature, return_confidence=False
                    )
                    u_map = entropy_active[indices_to_pred_active]
                else:
                    # Stats / Cosine Mode
                    conf_acc = None
                    if use_cosine:
                        # === 使用 Cosine 累积器 ===
                        # 窗口步数可以通过 self._get_confidence_window() 获取，或者直接全量
                        cosine_window = self._get_confidence_window()
                        if not cosine_window:
                             # 如果未指定窗口，默认使用全部步数（或者你可以指定一个默认值）
                             cosine_window = list(range(self.diffloss.gen_diffusion.num_timesteps))
                        conf_acc = CosineTrajectoryAccumulator(cosine_window)
                    elif use_semantic:
                        # === [Feature Semantic Consistency 实现核心] ===
                        
                        # 准备 Window (哪些步计算一致性)
                        semantic_window = self._get_confidence_window()
                        if not semantic_window:
                             semantic_window = list(range(self.diffloss.gen_diffusion.num_timesteps))
                        
                        # 提取出当前需要生成的那些位置对应的 LR 特征
                        target_lr_features = lr_tokens_active[indices_to_pred_active]
                        
                        conf_acc = SemanticConsistencyAccumulator(
                            target_features=target_lr_features,  # <--- 传入切片后的特征
                            window_steps=semantic_window
                        )
                    elif is_trajectory_mode:
                        conf_acc = TrajectoryConfidenceAccumulator(window_steps, sigma_delta=sigma_delta)
                    else:
                        conf_acc = VarianceConfidenceAccumulator(window_steps)
                    
                    sampled_token_latent, u_map_raw = self.diffloss.sample(
                        z_cond_active, temperature, confidence_accumulator=conf_acc, return_confidence=True
                    )
                    
                    # 标准化
                    if use_cosine or use_semantic:
                         u_map = u_map_raw # Cosine 不需要标准化 [0, 2]
                    elif is_trajectory_mode:
                         u_map = u_map_raw # Energy 不需要标准化
                    else:
                         u_map = standardize(u_map_raw, mu_u, sigma_u)

                if u_map is None or u_map.numel() == 0:
                    u_map = torch.zeros(z_cond_active.shape[0], device=z_cond_active.device)
                
                u_std = u_map

                # --- Stochastic Ranking ---
                noise = torch.randn_like(u_std)
                u_std = u_std + noise * stochastic_scale
                
                if is_main_process() and round_idx == 0:
                    print(f"DEBUG Score: Min={u_std.min():.3f}, Max={u_std.max():.3f}, Mean={u_std.mean():.3f}")
                u_std_flat = u_std.reshape(-1)
                num_masked = u_std_flat.shape[0]
                
                # --- 动态阈值筛选 ---
                progress_ratio = min(round_idx / ramp_steps, 1.0)
                current_threshold = start_threshold + (end_threshold - start_threshold) * progress_ratio
                
                pass_mask_flat = u_std_flat < current_threshold
                
                # Safety Net
                global_min_tokens = math.ceil(curr_bsz_active * self.seq_len * self.conf_pmin)
                safety_min = min(num_masked, max(1, global_min_tokens))
                
                if pass_mask_flat.sum().item() < safety_min:
                    topk = torch.topk(u_std_flat, k=safety_min, largest=False)
                    pass_mask_flat[topk.indices] = True
                
                fail_mask_flat = ~pass_mask_flat

                # 更新 Token
                pass_indices = (indices_to_pred_active[0][pass_mask_flat], indices_to_pred_active[1][pass_mask_flat])
                fail_indices = (indices_to_pred_active[0][fail_mask_flat], indices_to_pred_active[1][fail_mask_flat])
                
                cur_tokens_active = tokens_active.clone()
                cur_tokens_active[pass_indices] = sampled_token_latent[pass_mask_flat]
                
                if fail_mask_flat.any():
                    if self.remask_mode == "keep":
                        cur_tokens_active[fail_indices] = sampled_token_latent[fail_mask_flat]
                    else:
                        cur_tokens_active[fail_indices] = 0 
                
                mask_active[pass_indices] = 0
                if pass_mask_flat.any():
                    # 记录分数
                    acc_local_b = indices_to_pred_active[0][pass_mask_flat]
                    acc_t = indices_to_pred_active[1][pass_mask_flat]
                    acc_global_b = active_indices[acc_local_b]
                    saved_confidence[acc_global_b, acc_t] = u_map[pass_mask_flat]

                # === Remask逻辑 ===
                '''
                for i in range(curr_bsz_active):
                    global_idx = active_indices[i]
                    local_idx = i
                    current_mask_count = mask_active[local_idx].sum().item()
                    current_ratio = current_mask_count / num_hr_tokens
                    
                    if current_ratio < 0.05: 
                        continue
                    
                    known_mask = (mask_active[local_idx] == 0)
                    known_indices = known_mask.nonzero(as_tuple=True)[0]
                    
                    if len(known_indices) > num_remask:
                        scores = saved_confidence[global_idx, known_indices]
                        # 找出分数最高的 k 个 (Worst K)
                        topk = torch.topk(scores, k=num_remask, largest=True)
                        candidates_idx = known_indices[topk.indices]
                        mask_active[local_idx, candidates_idx] = 1
                '''

                tokens[active_indices] = cur_tokens_active
                mask[active_indices] = mask_active

                is_finished = (mask_active.sum(dim=-1) == 0)
                if is_finished.any():
                    finished_global_indices = active_indices[is_finished]
                    unfinished_batch_mask[finished_global_indices] = False

                if is_main_process():
                    accepted = pass_mask_flat.sum().item()
                    remaining_global = mask.sum().item()
                    print(
                        f"[Dynamic-MaskGIT] Round {round_idx}: "
                        f"Thresh {current_threshold:.2f}, "
                        f"Active {curr_bsz_active}/{bsz}, "
                        f"Accepted {accepted}, Remaining {remaining_global}"
                    )
                
                round_idx += 1
            actual_steps = round_idx
            
        tokens = self.unpatchify(tokens, shape=shape_hr)
        return tokens, actual_steps

def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=8, encoder_num_heads=8,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model