from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block, Mlp, DropPath

from models.diffloss import DiffLoss

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
# =========================================================================
# 自定义支持 RoPE 的 Transformer Block
# =========================================================================

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

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        # qkv: [3, B, num_heads, N, head_dim]
        # 注意这里 permute 成了 (2, 0, 3, 1, 4) -> [3, B, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # === 注入 RoPE (VARSR 风格) ===
        if freqs_cis is not None:
            # apply_rotary_emb 期望输入为 [B, L, H, D]
            # 当前 q, k 是 [B, H, L, D]，需要 permute
            q = q.transpose(1, 2) # -> [B, L, H, D]
            k = k.transpose(1, 2)
            q, k = apply_rotary_emb(q, k, freqs_cis)
            q = q.transpose(1, 2) # -> [B, H, L, D]
            k = k.transpose(1, 2)
        # ============================

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RoPEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # 使用自定义的 RoPEAttention
        self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs_cis=None):
        # 将 freqs_cis 传入 Attention
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
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
                 #label_drop_prob=0.1,
                 #class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,   #这里需要设置为LR转码后的长度
                 diffloss_d=6,      #显存紧张可以降低
                 diffloss_w=1024,   #可以与embed_dim保持一致
                 num_sampling_steps='100',
                 diffusion_batch_mul=1,         #单卡最好设为1，之前为4
                 grad_checkpointing=False,
                 mse_weight=0.25,
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
        #self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        # Transformer Blocks：堆叠 16 层
        # self.encoder_blocks = nn.ModuleList([
        #     Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
        #           proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_blocks = nn.ModuleList([
            RoPEBlock(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                    drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics

        # 投影层：如果 Encoder 和 Decoder 维度不一样，这里负责转换 (通常是一样的)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        # mask_token：这是一个特殊的向量，代表“未知”，所有被遮住的位置，都会填入这个向量
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Decoder 位置编码
        #self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        # Transformer Blocks：堆叠 16 层
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
        #           proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        self.decoder_blocks = nn.ModuleList([
            RoPEBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                    drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

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

        

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        self.apply(self._init_weights)
        #初始化投影层
        torch.nn.init.xavier_uniform_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            torch.nn.init.constant_(self.final_proj.bias, 0)



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
    
    def forward_mae_encoder(self, x_hr, mask, x_lr, shape_hr, shape_lr):
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
        # [修改] 移除中间定义的 pos_embed 变量和加法
        x = torch.cat([lr_embedding, hr_embedding], dim=1)

        # 计算完整的 RoPE 频率
        freqs_cis_full = self.get_rope_freqs(shape_hr, shape_lr, x.device, head_dim)
        freqs_cis_full = freqs_cis_full.unsqueeze(0).repeat(bsz, 1, 1) # 扩展到 Batch

        # 构造 Mask (LR 永远可见)
        num_lr_tokens = shape_lr[0] * shape_lr[1]
        mask_with_buffer = torch.cat([torch.zeros(bsz, num_lr_tokens, device=x.device), mask], dim=1)
        
        x = self.z_proj_ln(x)

        # Dropping (同时筛选 Feature 和 频率)
        keep_indices = (1 - mask_with_buffer).nonzero(as_tuple=True)
        
        # 计算保留的 token 数量 (假设 batch 内一致，通常 MAR 设计也是如此)
        num_kept = int(mask_with_buffer.shape[1] - mask_with_buffer.sum(dim=1)[0].item())
        
        # 筛选 Feature
        x = x[keep_indices].reshape(bsz, num_kept, embed_dim)
        
        # [新增] 筛选 Frequencies
        freqs_cis_kept = freqs_cis_full[keep_indices].reshape(bsz, num_kept, -1)

        # Apply Transformer Blocks (传入 freqs_cis)
        for block in self.encoder_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():         
                # [修改] 传入 freqs_cis=freqs_cis_kept
                x = checkpoint(block, x, freqs_cis_kept)
            else:
                x = block(x, freqs_cis=freqs_cis_kept)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask, shape_hr, shape_lr):
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

        # 计算完整 RoPE 频率
        freqs_cis_full = self.get_rope_freqs(shape_hr, shape_lr, x.device, head_dim)
        # 注意：这里不需要 repeat 到 batch 维度，apply_rotary_emb 会处理广播，
        # 或者为了保险起见，跟 Encoder 保持一致：
        freqs_cis_full = freqs_cis_full.repeat(bsz, 1, 1) # 可选，视显存情况而定

        # Apply Transformer Blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                # [修改] 传入 freqs_cis
                x = checkpoint(block, x, freqs_cis_full)
        else:
            for block in self.decoder_blocks:
                x = block(x, freqs_cis=freqs_cis_full)
        x = self.decoder_norm(x)

        # 移除 Buffer (LR tokens)
        x = x[:, num_lr_tokens:]

        return x

    def forward_loss(self, z, target, mask):
        # z: Decoder 预测出的 Latent 特征 [Batch, Seq_Len, Dim]
        # target: 真实的 Latent 特征 (Ground Truth) [Batch, Seq_Len, Dim]
        # mask: 当前的遮挡掩码 [Batch, Seq_Len]  
        z_projected = self.final_proj(z)
        loss_mse_element = (z_projected - target) ** 2

        loss_mse_token = loss_mse_element.mean(dim=-1)

        loss_mse = (loss_mse_token * mask).sum() / (mask.sum() + 1e-6)

        bsz, seq_len, _ = target.shape

        # 每一张图片在一次 Forward 中同时学习 4 个不同的扩散时间步（diffusion_batch_mul）
        target_diff = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z_diff = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask_diff = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)


        # z 是条件 (Condition)，target 是要加噪的数据 (x_start)
        loss_diff = self.diffloss(z=z_diff, target=target_diff, mask=mask_diff)

        loss = loss_diff + self.mse_weight * loss_mse
        
        return loss

    def forward(self, x_hr, x_lr):
        # x_hr: HR Latents [B, 16, 16, 16]
        # x_lr: LR Latents [B, 16, 32, 32]
        # class embed
        #class_embedding = self.class_emb(x_lr)

        # 显式获取 Latent 的真实高宽
        # 这就是支持长方形输入的关键：不再假设 H=W
        #_, _, h_hr, w_hr = x_hr.shape
        #_, _, h_lr, w_lr = x_lr.shape
        shape_hr = (self.seq_h, self.seq_w)
        shape_lr = (self.seq_h, self.seq_w)

        # patchify and mask (drop) tokens
        # 切片化
        lr_tokens = self.patchify(x_lr)
        hr_tokens = self.patchify(x_hr)
        # 获取当前 batch 的真实 token 数量
        num_tokens = hr_tokens.shape[1]
        # 备份groundtruth（gt_latents）
        gt_latents = hr_tokens.clone().detach()
        # 生成随机掩码
        orders = self.sample_orders(bsz=hr_tokens.size(0), num_tokens = num_tokens)
        mask = self.random_masking(hr_tokens, orders)

        # mae encoder
        x = self.forward_mae_encoder(hr_tokens, mask, lr_tokens, shape_hr, shape_lr)
        # mae decoder
        z = self.forward_mae_decoder(x, mask, shape_hr, shape_lr)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", x_lr=None, temperature=1.0, progress=False, target_seq_len=None):
        #num_iter：自回归迭代次数（官方推荐设置256）
        # 必须有 LR 输入
        if x_lr is None:
            raise ValueError("Super-Resolution requires LR input!")
        
        shape_lr = (self.seq_h, self.seq_w)
        shape_hr = (self.seq_h, self.seq_w)
        num_hr_tokens = self.seq_len
        # 先展平 LR tokens
        lr_tokens = self.patchify(x_lr)

        # if target_seq_len is not None:
        #     current_area = h_lr * w_lr
        #     scale = (target_seq_len / current_area) ** 0.5
            
        #     h_hr = int(round(h_lr * scale))
        #     w_hr = int(round(w_lr * scale))
            
        #     if h_hr * w_hr != target_seq_len:
        #          # 回退策略：如果是正方形任务，直接开方
        #          if h_lr == w_lr:
        #              side = int(target_seq_len**0.5)
        #              h_hr, w_hr = side, side
        #          else:
        #              print(f"Warning: Target seq len {target_seq_len} does not match aspect ratio of LR {h_lr}x{w_lr}. Shape inference might be inaccurate.")
        # else:
        #     # 默认逻辑：按照 MAR 的标准设定，Latent 边长放大 4 倍 (Area x16)
        #     h_hr = h_lr * 4
        #     w_hr = w_lr * 4

        # shape_hr = (h_hr, w_hr)
        # num_hr_tokens = h_hr * w_hr
        # init and sample generation orders
        # 初始化掩码：全为 1 (代表全图被遮挡/未知)
        mask = torch.ones(bsz, num_hr_tokens).cuda()
        # 初始化 Token：全为 0 (画布是黑的)
        tokens = torch.zeros(bsz, num_hr_tokens, self.token_embed_dim).cuda()
        # 生成随机顺序：决定先画哪儿，后画哪儿
        #orders = self.sample_orders(bsz)

        # 直接调用修改后的 sample_orders，不用手写循环了
        orders = self.sample_orders(bsz, num_tokens=num_hr_tokens)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, lr_tokens, shape_hr, shape_lr)
            
            # mae decoder
            z = self.forward_mae_decoder(x, mask, shape_hr, shape_lr)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            #mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
            mask_len = torch.Tensor([np.floor(num_hr_tokens * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, num_hr_tokens) 
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            ##469-479为测试时注释掉的部分，需要恢复，480,481需要删除

            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (num_hr_tokens - mask_len[0]) / num_hr_tokens
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
            #z = self.final_proj(z)
            #sampled_token_latent = z

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
        
        # unpatchify
        tokens = self.unpatchify(tokens, shape=shape_hr)
        return tokens


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
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
