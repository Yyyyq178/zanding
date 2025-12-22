import torch
import torch.nn.functional as F


def _sobel_kernels(device, dtype):
    kernel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=dtype
    )
    kernel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=dtype
    )
    return kernel_x, kernel_y


def sobel_edges(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Sobel edge magnitude per-channel.

    Args:
        x: [B, C, H, W]
    Returns:
        edges: [B, C, H, W]
    """
    bsz, channels, _, _ = x.shape
    kernel_x, kernel_y = _sobel_kernels(x.device, x.dtype)
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)
    return edges


def normalize_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each image to [0, 1] with per-image min-max."""
    bsz = x.shape[0]
    x_flat = x.view(bsz, -1)
    x_min = x_flat.min(dim=1, keepdim=True).values
    x_max = x_flat.max(dim=1, keepdim=True).values
    x_norm = (x_flat - x_min) / (x_max - x_min + eps)
    return x_norm.view_as(x)


def build_degradation_map(hr: torch.Tensor, lr: torch.Tensor, w_pix: float, w_grad: float) -> torch.Tensor:
    """Build pixel-level degradation map from HR/LR pairs.

    Returns:
        D_pix_gt: [B, 1, H, W] in [0, 1]
    """
    e_pix = (hr - lr).abs().mean(dim=1, keepdim=True)
    hr_edges = sobel_edges(hr)
    lr_edges = sobel_edges(lr)
    e_grad = (hr_edges - lr_edges).abs().mean(dim=1, keepdim=True)
    d_pix = w_pix * e_pix + w_grad * e_grad
    return normalize_per_image(d_pix)


def get_token_grid_size(model) -> tuple:
    """Infer token grid size (H, W) from model metadata."""
    model_ref = model.module if hasattr(model, "module") else model
    if hasattr(model_ref, "seq_h") and hasattr(model_ref, "seq_w"):
        return model_ref.seq_h, model_ref.seq_w
    if hasattr(model_ref, "patch_embed") and hasattr(model_ref.patch_embed, "grid_size"):
        return model_ref.patch_embed.grid_size
    if hasattr(model_ref, "num_patches"):
        grid = int(model_ref.num_patches ** 0.5)
        return grid, grid
    raise AttributeError("Unable to infer token grid size from model.")


def pool_degradation_to_tokens(d_pix: torch.Tensor, model) -> torch.Tensor:
    """Pool pixel-level degradation map to token grid.

    Args:
        d_pix: [B, 1, H, W]
    Returns:
        d_tok: [B, N]
    """
    grid_h, grid_w = get_token_grid_size(model)
    _, _, h, w = d_pix.shape
    if h % grid_h == 0 and w % grid_w == 0:
        kernel_h, kernel_w = h // grid_h, w // grid_w
        d_tok = F.avg_pool2d(d_pix, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
    else:
        d_tok = F.adaptive_avg_pool2d(d_pix, output_size=(grid_h, grid_w))

    d_tok = d_tok.flatten(1)
    return d_tok