#定义了 MAR 的主体架构
from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 每个位置的编码维度
    pos: 位置坐标列表 (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), 外积

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # 使用一半的维度编码 H，一半的维度编码 W
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size_hr, grid_size_lr):
    """
    为 HR 和 LR 生成对齐的 2D sin-cos 位置编码
    grid_size_hr: int, 例如 32 (对应 512分辨率)
    grid_size_lr: int, 例如 8  (对应 128分辨率)
    """
    # 1. 生成 HR 的坐标网格 (0 到 grid_size_hr)
    grid_h = np.arange(grid_size_hr, dtype=np.float32)
    grid_w = np.arange(grid_size_hr, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w 在前
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_hr, grid_size_hr])
    
    # 生成 HR 的 Embedding
    pos_embed_hr = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # 2. 生成 LR 的坐标网格 (关键步骤！)
    # 我们使用 linspace 将 LR 的坐标“拉伸”覆盖到和 HR 一样的空间范围
    # 例如: LR 的 0~7 坐标会被映射到 HR 的 0~31 范围内
    grid_l_h = np.linspace(0, grid_size_hr - 1, grid_size_lr, dtype=np.float32)
    grid_l_w = np.linspace(0, grid_size_hr - 1, grid_size_lr, dtype=np.float32)
    grid_l = np.meshgrid(grid_l_w, grid_l_h)
    grid_l = np.stack(grid_l, axis=0)
    grid_l = grid_l.reshape([2, 1, grid_size_lr, grid_size_lr])
    
    # 生成 LR 的 Embedding
    pos_embed_lr = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    
    # 3. 拼接: LR 在前 (Buffer), HR 在后
    # 结果 shape: [seq_len + buffer_size, embed_dim]
    pos_embed = np.concatenate([pos_embed_lr, pos_embed_hr], axis=0)
    
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

        # --------------------------------------------------------------------------
        # Class Embedding
        #self.num_classes = class_num
        #self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        #self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        #self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics

        # 投影层：把 VAE 的 16 维向量映射到 Transformer 的 1024 维
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        # 缓冲区大小：定义前缀长度 (64)
        self.buffer_size = buffer_size
        # 位置编码：这是一个可学习的参数表，长度 = 序列长度(256) + 缓冲区长度(64) = 320，维度 = 1024
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        # Transformer Blocks：堆叠 16 层
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics

        # 投影层：如果 Encoder 和 Decoder 维度不一样，这里负责转换 (通常是一样的)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        # mask_token：这是一个特殊的向量，代表“未知”，所有被遮住的位置，都会填入这个向量
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Decoder 位置编码
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        # Transformer Blocks：堆叠 16 层
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Diffusion 位置编码：这是给 DiffLoss 用的额外位置信息
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

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

        self.initialize_weights()

    def initialize_weights(self):
        # parameters
        #torch.nn.init.normal_(self.class_emb.weight, std=.02)
        #torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        grid_size_hr = int(self.seq_len**0.5) 
        grid_size_lr = int(self.buffer_size**0.5)

        pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed_learned.shape[-1], # embed_dim (e.g. 768 or 1024)
            grid_size_hr, 
            grid_size_lr
        )
        # 赋值给 Encoder 和 Decoder
        # unsqueeze(0) 是为了增加 Batch 维度: [1, Total_Len, Dim]
        self.encoder_pos_embed_learned.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed_learned.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 赋值给 Diffusion (只取 HR 部分，去掉 buffer)
        pos_embed_hr_only = pos_embed[self.buffer_size:, :] 
        self.diffusion_pos_embed_learned.data.copy_(torch.from_numpy(pos_embed_hr_only).float().unsqueeze(0))

        # 3. 初始化通用层 (Linear, LayerNorm)
        # 这会递归初始化包括 DiffLoss 在内的所有层
        self.apply(self._init_weights)
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

    def unpatchify(self, x):
        # 反向重塑
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
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

    def forward_mae_encoder(self, x_hr, mask, x_lr):
        # x: [Batch, Seq_Len=256, Dim=16] (VAE 输出的 token)
        # mask: [Batch, Seq_Len=256] (0=可见, 1=遮挡)
        # x_lr: LR tokens（来自VAE编码，维度 16）
        hr_embedding = self.z_proj(x_hr)
        lr_embedding = self.z_proj(x_lr)
        bsz, seq_len, embed_dim = hr_embedding.shape

        # concat buffer
        # x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        x = torch.cat([lr_embedding, hr_embedding], dim=1)
        # 给buffer打上0，表示永远可见
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # 注入类别条件 (CFG 技巧)
        #if self.training:
            # 训练时有 10% 概率把类别 Embedding 替换成“假标签”(fake_latent)
            #drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            #drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            #class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        # 把类别向量填满那 64 个 buffer 空位（重点修改这一步）
        #x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        # 这一步赋予了每个 token 空间位置信息
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        # 只取出 mask=0 的位置。这一步极大地减少了计算量
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        # 此时 x 的长度变短了 (比如从 320 变成了 64+几十个可见token)
        for block in self.encoder_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():         
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):
        # x: Encoder 的输出 (只有可见部分 + Buffer)
        # mask: 原始的遮挡掩码
        x = self.decoder_embed(x)
        # 重建带有 buffer 的完整 mask
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens（填补空缺）
        # 先造一个全是不可见的底板，形状是完整的[Batch, 320, dim]
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        # 把可见特征填回它原来的位置
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        # Decoder 处理的是完整的长序列 (320)
        # 它要利用可见信息，去“猜” mask_token 那个位置应该是什么
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        # 移除buffer
        x = x[:, self.buffer_size:]
        # 加上diffusion位置编码
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        # z: Decoder 预测出的 Latent 特征 [Batch, Seq_Len, Dim]
        # target: 真实的 Latent 特征 (Ground Truth) [Batch, Seq_Len, Dim]
        # mask: 当前的遮挡掩码 [Batch, Seq_Len]  
        bsz, seq_len, _ = target.shape
        # 每一张图片在一次 Forward 中同时学习 4 个不同的扩散时间步（diffusion_batch_mul）
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        # z 是条件 (Condition)，target 是要加噪的数据 (x_start)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, x_hr, x_lr):
        # x_hr: HR Latents [B, 16, 16, 16]
        # x_lr: LR Latents [B, 16, 32, 32]
        # class embed
        #class_embedding = self.class_emb(x_lr)

        # patchify and mask (drop) tokens
        # 切片化
        lr_tokens = self.patchify(x_lr)
        hr_tokens = self.patchify(x_hr)
        # 备份groundtruth（gt_latents）
        gt_latents = hr_tokens.clone().detach()
        # 生成随机掩码
        orders = self.sample_orders(bsz=hr_tokens.size(0))
        mask = self.random_masking(hr_tokens, orders)

        # mae encoder
        x = self.forward_mae_encoder(hr_tokens, mask, lr_tokens)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", x_lr=None, temperature=1.0, progress=False):
        #num_iter：自回归迭代次数（官方推荐设置256）
        # init and sample generation orders
        # 初始化掩码：全为 1 (代表全图被遮挡/未知)
        mask = torch.ones(bsz, self.seq_len).cuda()
        # 初始化 Token：全为 0 (画布是黑的)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        # 生成随机顺序：决定先画哪儿，后画哪儿
        orders = self.sample_orders(bsz)

        # 如果推理时传入了 x_lr（Latent形式）,先把它变成tokens
        if x_lr is not None:
            lr_tokens = self.patchify(x_lr)
        else:
            raise ValueError("Super-Resolution requires LR input!")

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            # 如果启用 CFG (cfg != 1.0)，我们需要构造“双倍 Batch”
            #if x_lr is not None:
            #    class_embedding = self.class_emb(x_lr)
            #else:
            #    class_embedding = self.fake_latent.repeat(bsz, 1)
            #if not cfg == 1.0:
            #    tokens = torch.cat([tokens, tokens], dim=0)
            #    class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
            #    mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, lr_tokens)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
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
