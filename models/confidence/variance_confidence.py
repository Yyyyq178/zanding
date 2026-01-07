import json
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

class CosineTrajectoryAccumulator:
    """
    方案4实现：基于 x0 轨迹方向一致性的置信度 (Cosine Similarity)。
    
    逻辑：
    1. 计算每一步的更新向量: delta_t = x0_current - x0_prev
    2. 计算相邻两个更新向量的余弦相似度: sim = CosineSim(delta_t, delta_prev)
    3. 定义不确定性 (Uncertainty) = 1 - sim
       (sim 越接近 1，表示方向一致，不确定性越低；sim 越小，表示方向抖动，不确定性越高)
    """
    def __init__(self, window_steps: Sequence[int], eps: float = 1e-6):
        self.window_steps: Set[int] = set(int(s) for s in window_steps)
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.u_accum = None    # 累积的不确定性分数
        self.x0_prev = None    # 上一步预测的 x0
        self.delta_prev = None # 上一步的更新向量
        self.cnt = 0

    def update(self, t: int, x_start: torch.Tensor, **kwargs) -> None:
        """
        Args:
            t: 当前 timestep
            x_start: 当前预测的 x0 [N, C] 或 [B, C, H, W]
        """
        if t not in self.window_steps:
            return

        # 自动探测维度 (Token模式 或 Image模式)
        ndim = x_start.ndim
        
        # === 初始化累积器 ===
        if self.u_accum is None:
            if ndim == 2:
                # Token模式: [N, C] -> 输出分数 [N]
                N, C = x_start.shape
                self.u_accum = torch.zeros(N, device=x_start.device, dtype=x_start.dtype)
            elif ndim == 4:
                # Image模式: [B, C, H, W] -> 输出分数 [B, H, W]
                B, C, H, W = x_start.shape
                self.u_accum = torch.zeros(B, H, W, device=x_start.device, dtype=x_start.dtype)
            
            # 第一步没有 delta，只记录 x0
            self.x0_prev = x_start.detach().clone()
            return

        # === 计算当前步的 Delta ===
        # current_delta 指向当前预测值的方向变化
        current_delta = x_start - self.x0_prev
        
        # === 计算余弦相似度并累积 ===
        if self.delta_prev is not None:
            # 计算 Cosine Similarity
            # dim=1 是 Channel 维度 (无论是 [N, C] 还是 [B, C, H, W])
            sim = F.cosine_similarity(current_delta, self.delta_prev, dim=1, eps=self.eps)
            
            # 将 Similarity 转换为 Uncertainty (越小越好)
            # Range: [0, 2] (Sim: 1 -> Unc: 0; Sim: -1 -> Unc: 2)
            uncertainty = 1.0 - sim
            
            self.u_accum += uncertainty
            self.cnt += 1
        
        # === 更新状态 ===
        self.delta_prev = current_delta.detach().clone()
        self.x0_prev = x_start.detach().clone()

    def finalize(self) -> torch.Tensor:
        """
        返回平均不确定性 u_i
        """
        if self.u_accum is None or self.cnt == 0:
            return torch.zeros(1)
        
        # 平均值
        return self.u_accum / max(self.cnt, 1)
    
class TrajectoryConfidenceAccumulator:
    """
    MD方案实现：x0 轨迹差分能量累积器 (Trajectory Stability)
    兼容 2D (Tokens: [N, C]) 和 4D (Images: [B, C, H, W]) 输入。
    """
    def __init__(self, window_steps: Sequence[int], sigma_delta: Optional[torch.Tensor] = None, eps: float = 1e-6):
        self.window_steps: Set[int] = set(int(s) for s in window_steps)
        self.sigma_delta = sigma_delta # 初始传入可能是 [1, C, 1, 1]
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.u_accum = None
        self.x0_prev = None
        self.cnt = 0

    def update(self, t: int, x_start: torch.Tensor, **kwargs) -> None:
        """
        Args:
            t: 当前 timestep
            x_start: 当前预测的 x0。可能是 [N, C] 或 [B, C, H, W]
        """
        if t not in self.window_steps:
            return

        # 自动探测维度
        ndim = x_start.ndim
        
        # === 初始化 ===
        if self.u_accum is None:
            if ndim == 2:
                # Token模式: [N, C] -> 输出分数 [N]
                N, C = x_start.shape
                self.u_accum = torch.zeros(N, device=x_start.device, dtype=x_start.dtype)
            elif ndim == 4:
                # 图片模式: [B, C, H, W] -> 输出分数 [B, H, W]
                B, C, H, W = x_start.shape
                self.u_accum = torch.zeros(B, H, W, device=x_start.device, dtype=x_start.dtype)
            else:
                raise ValueError(f"Unsupported x_start shape: {x_start.shape}")
            
            self.x0_prev = x_start.detach().clone()
            return

        # === 累积计算 ===
        if self.x0_prev is not None:
            # 1. 计算差分
            delta = x_start - self.x0_prev
            
            # 2. 归一化 (Sigma)
            if self.sigma_delta is not None:
                # 确保设备一致
                if self.sigma_delta.device != delta.device:
                    self.sigma_delta = self.sigma_delta.to(delta.device)
                
                # 动态调整 Sigma 形状以匹配输入
                # 先把 sigma 展平成向量 [C]
                sigma_flat = self.sigma_delta.flatten()
                
                if ndim == 2:
                    # Input: [N, C] -> Sigma: [1, C]
                    denom = sigma_flat.view(1, -1)
                elif ndim == 4:
                    # Input: [B, C, H, W] -> Sigma: [1, C, 1, 1]
                    denom = sigma_flat.view(1, -1, 1, 1)
                
                delta = delta / (denom + self.eps)
            
            # 3. 计算能量 (平方和)
            # 无论是 [N, C] 还是 [B, C, H, W]，Channel 都在 dim=1
            u_step = (delta ** 2).mean(dim=1)
            
            # 4. 累积
            self.u_accum += u_step
            self.cnt += 1
            self.x0_prev = x_start.detach().clone()

    def finalize(self) -> torch.Tensor:
        """
        返回平均能量 u_i
        """
        if self.u_accum is None or self.cnt == 0:
            # 防止空数据返回，返回一个标量0即可，外部通常会处理形状
            return torch.zeros(1)
        
        # 平均能量
        return self.u_accum / max(self.cnt, 1)
    

class VarianceConfidenceAccumulator:
    """
    Lightweight accumulator that gathers variance confidence statistics during sampling.

    Accumulates mean log-variance over the channel dimension for a configurable
    set of timesteps and averages across the collected steps at the end.
    """

    def __init__(self, window_steps: Sequence[int], eps: float = 1e-8, collect_conf_stats: bool = False):
        self.window_steps: Set[int] = set(int(s) for s in window_steps)
        self.eps = eps
        self.collect_conf_stats = collect_conf_stats
        self.reset()

    def reset(self, shape_like: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the accumulator.

        Args:
            shape_like: a tensor whose spatial shape matches the target tokens.
        """
        if shape_like is None:
            self.u_accum = None
        else:
            # Expect shape [B, C, H, W]; reduce to [B, H, W] when used.
            spatial_shape = shape_like.shape[2:]
            device = shape_like.device
            dtype = shape_like.dtype
            self.u_accum = torch.zeros(shape_like.shape[0], *spatial_shape, device=device, dtype=dtype)
        self.cnt = 0
        if self.collect_conf_stats:
            self._per_step = []

    def maybe_add(self, t: int, log_variance: torch.Tensor) -> None:
        """
        Add log-variance statistics for a given timestep when it falls into the window.

        Args:
            t: current timestep (int).
            log_variance: tensor shaped [B, C, H, W].
        """
        if t not in self.window_steps:
            return
        # Mean over channel dimension to [B, H, W]
        mean_logvar = log_variance.mean(dim=1)
        if self.u_accum is None:
            self.reset(shape_like=log_variance)
        self.u_accum = self.u_accum + mean_logvar
        self.cnt += 1
        if self.collect_conf_stats:
            self._per_step.append(mean_logvar.detach().cpu())
    def update(self, t: int, log_variance: Optional[torch.Tensor] = None, **kwargs) -> None:
        if log_variance is None: return
        if t not in self.window_steps: return
        
        mean_logvar = log_variance.mean(dim=1)
        if self.u_accum is None:
            self.reset(shape_like=log_variance)
        self.u_accum = self.u_accum + mean_logvar
        self.cnt += 1
    def finalize(self) -> torch.Tensor:
        """
        Finalize aggregation and return the averaged u_map.

        Returns:
            Tensor shaped [B, H, W] (or zeros if nothing was accumulated).
        """
        if self.u_accum is None or self.cnt == 0:
            return torch.zeros(0)
        return self.u_accum / max(self.cnt, 1)

    def dump_stats(self, path: Path) -> None:
        """Persist per-step stats for analysis; noop if not collecting."""
        if not self.collect_conf_stats or not self._per_step:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stack to shape [num_steps, B, H, W]
        arr = torch.stack(self._per_step, dim=0).numpy()
        np.savez_compressed(path, per_step_u=arr)


def standardize(u_map: torch.Tensor, mu_u: torch.Tensor, sigma_u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize the confidence map.

    Args:
        u_map: raw confidence map.
        mu_u: precomputed mean (broadcastable).
        sigma_u: precomputed std (broadcastable).
        eps: numerical stability term.
    """
    return (u_map - mu_u) / (sigma_u + eps)


def load_confidence_stats(path: Optional[str]) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None]:
    """
    加载统计数据。
    如果是旧版 .npz，返回 (mu, sigma)。
    如果是新版 (trajectory)，返回 sigma_delta。
    """
    if path is None or path == "":
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"Warning: Stats file not found {path}")
        return None
        
    data = np.load(path_obj)
    
    # 检测是哪种统计文件
    if "sigma_delta" in data:
        # 新方案：只返回 sigma_delta [C]
        sigma_delta = torch.from_numpy(data["sigma_delta"])
        # 调整形状为 [1, C, 1, 1] 以便广播
        return sigma_delta.view(1, -1, 1, 1)
    elif "mu_u" in data:
        # 旧方案
        mu_u = torch.from_numpy(data["mu_u"])
        sigma_u = torch.from_numpy(data["sigma_u"])
        return mu_u, sigma_u
    else:
        return None