import json
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

import numpy as np
import torch


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


def load_confidence_stats(path: Optional[str]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load (mu_u, sigma_u) from a file. Supports npz, json, or yaml (via pyyaml if installed).

    Returns tensors or None if path is not provided.
    """
    if path is None or path == "":
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Confidence stats file not found: {path}")
    suffix = path_obj.suffix.lower()
    if suffix == ".npz":
        data = np.load(path_obj)
        mu_u = torch.from_numpy(data["mu_u"])
        sigma_u = torch.from_numpy(data["sigma_u"])
    elif suffix in {".json"}:
        with open(path_obj, "r") as f:
            data = json.load(f)
        mu_u = torch.tensor(data["mu_u"])
        sigma_u = torch.tensor(data["sigma_u"])
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("pyyaml is required to load YAML confidence stats") from exc
        with open(path_obj, "r") as f:
            data = yaml.safe_load(f)
        mu_u = torch.tensor(data["mu_u"])
        sigma_u = torch.tensor(data["sigma_u"])
    else:
        raise ValueError(f"Unsupported confidence stats format: {suffix}")
    return mu_u, sigma_u