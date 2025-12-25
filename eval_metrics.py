#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline evaluator (HR folder + SR folder) with metric computation consistent with engine_mar.py:

- All metrics are created via pyiqa.create_metric(...)
- Full-reference metrics use (SR, HR): psnr, ssim, lpips, dists, ms_ssim
- No-reference metrics use (SR): musiq, brisque, maniqa, clipiqa, niqe, piqe
- FID is computed directly on the two input folders: fid(sr_dir, hr_dir)
  (no intermediate sr_images/hr_images saving)

Pair matching:
- Match by relative path WITHOUT extension (robust across different extensions).

Distributed:
- If launched via torchrun and torch.distributed is initialized:
  - Full/No-ref metrics are computed on per-rank shards then reduced (SUM) to global mean.
  - FID is computed once on rank0 from the full input folders and broadcast.

Example (single GPU):
  python eval_offline_direct.py --hr_dir /path/HR --sr_dir /path/SR --device cuda:0 --align resize_sr_to_hr

Example (multi-GPU):
  torchrun --nproc_per_node=4 eval_offline_direct.py --hr_dir ... --sr_dir ... --device cuda --align resize_sr_to_hr

Dependencies:
  pip install torch torchvision pillow tqdm pyiqa
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import pyiqa

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def broadcast(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        dist.broadcast(t, src=src)
    return t


def pil_load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor_01(img: Image.Image) -> torch.Tensor:
    return TF.to_tensor(img).float()


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _collect_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and _is_image(p)]


def _build_key_map(root: Path) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in sorted(_collect_images(root)):
        rel = p.relative_to(root)
        key = str(rel.with_suffix("")).replace("\\", "/")
        if key not in m:
            m[key] = p
    return m


@dataclass
class Pair:
    key: str
    hr: Path
    sr: Path


def match_pairs(hr_dir: Path, sr_dir: Path) -> List[Pair]:
    hr_map = _build_key_map(hr_dir)
    sr_map = _build_key_map(sr_dir)
    keys = sorted(set(hr_map.keys()).intersection(sr_map.keys()))
    return [Pair(k, hr_map[k], sr_map[k]) for k in keys]


class PairedDataset(Dataset):
    def __init__(self, pairs: List[Pair], align: str):
        self.pairs = pairs
        self.align = align

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        hr_img = pil_load_rgb(p.hr)
        sr_img = pil_load_rgb(p.sr)

        if self.align == "resize_sr_to_hr":
            if sr_img.size != hr_img.size:
                sr_img = sr_img.resize(hr_img.size, resample=Image.BICUBIC)
        elif self.align == "resize_both_to_min":
            if sr_img.size != hr_img.size:
                w = min(sr_img.size[0], hr_img.size[0])
                h = min(sr_img.size[1], hr_img.size[1])
                sr_img = sr_img.resize((w, h), resample=Image.BICUBIC)
                hr_img = hr_img.resize((w, h), resample=Image.BICUBIC)
        elif self.align == "center_crop_to_min":
            if sr_img.size != hr_img.size:
                w = min(sr_img.size[0], hr_img.size[0])
                h = min(sr_img.size[1], hr_img.size[1])
                sr_img = TF.center_crop(sr_img, [h, w])
                hr_img = TF.center_crop(hr_img, [h, w])
        elif self.align == "none":
            pass
        else:
            raise ValueError(f"Unknown align mode: {self.align}")

        hr = pil_to_tensor_01(hr_img).clamp(0, 1)
        sr = pil_to_tensor_01(sr_img).clamp(0, 1)
        return {"key": p.key, "hr": hr, "sr": sr}


def shard_indices(n: int, rank: int, world_size: int) -> List[int]:
    return list(range(rank, n, world_size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--sr_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--align",
        type=str,
        default="none",
        choices=["none", "resize_sr_to_hr", "resize_both_to_min", "center_crop_to_min"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--no_fid", action="store_true")
    args = parser.parse_args()

    hr_dir = Path(args.hr_dir)
    sr_dir = Path(args.sr_dir)
    assert hr_dir.is_dir(), f"HR dir not found: {hr_dir}"
    assert sr_dir.is_dir(), f"SR dir not found: {sr_dir}"

    rank = get_rank()
    world_size = get_world_size()

    pairs = match_pairs(hr_dir, sr_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if rank == 0:
        print(f"Matched pairs: {len(pairs)}")
        if len(pairs) == 0:
            raise RuntimeError("No matched pairs found (match by relative path without extension).")
    barrier()

    shard = [pairs[i] for i in shard_indices(len(pairs), rank, world_size)]
    dl = DataLoader(
        PairedDataset(shard, align=args.align),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=False,
    )

    device = torch.device(args.device)

    metric_specs = {
        "psnr": ("psnr", True),
        "ssim": ("ssim", True),
        "lpips": ("lpips", True),
        "dists": ("dists", True),
        "ms_ssim": ("ms_ssim", True),
        "musiq": ("musiq", False),
        "brisque": ("brisque", False),
        "maniqa": ("maniqa", False),
        "clipiqa": ("clipiqa", False),
        "niqe": ("niqe", False),
        "piqe": ("piqe", False),
    }

    full_ref_metrics: Dict[str, torch.nn.Module] = {}
    no_ref_metrics: Dict[str, torch.nn.Module] = {}
    for log_name, (metric_name, is_full_ref) in metric_specs.items():
        m = pyiqa.create_metric(metric_name, device=device)
        try:
            m.eval()
        except Exception:
            pass
        (full_ref_metrics if is_full_ref else no_ref_metrics)[log_name] = m

    fid_metric = None
    if not args.no_fid:
        fid_metric = pyiqa.create_metric("fid", device=device)
        try:
            fid_metric.eval()
        except Exception:
            pass

    sums = {k: 0.0 for k in metric_specs.keys()}
    counts = {k: 0 for k in metric_specs.keys()}

    for batch in tqdm(dl, desc=f"Rank{rank} metrics", disable=(rank != 0)):
        sr = batch["sr"].to(device, non_blocking=True).float().clamp(0, 1)
        hr = batch["hr"].to(device, non_blocking=True).float().clamp(0, 1)

        if sr.shape[-2:] != hr.shape[-2:]:
            raise RuntimeError(
                f"Size mismatch with align='{args.align}': SR {tuple(sr.shape[-2:])} vs HR {tuple(hr.shape[-2:])}."
            )

        with torch.no_grad():
            for name, metric in full_ref_metrics.items():
                v = metric(sr, hr)
                v = v.mean() if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=device)
                sums[name] += float(v.detach().cpu()) * sr.size(0)
                counts[name] += sr.size(0)

            for name, metric in no_ref_metrics.items():
                v = metric(sr)
                v = v.mean() if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=device)
                sums[name] += float(v.detach().cpu()) * sr.size(0)
                counts[name] += sr.size(0)

    results: Dict[str, Optional[float]] = {}
    for k in metric_specs.keys():
        s = torch.tensor([sums[k]], device=device, dtype=torch.float64)
        c = torch.tensor([counts[k]], device=device, dtype=torch.float64)
        all_reduce_sum(s)
        all_reduce_sum(c)
        results[k] = float(s.item() / c.item()) if float(c.item()) > 0 else None

    if not args.no_fid:
        fid_val = torch.tensor([float("nan")], device=device, dtype=torch.float64)
        if rank == 0:
            try:
                fid_val = torch.tensor([float(fid_metric(str(sr_dir), str(hr_dir)))], device=device, dtype=torch.float64)
            except Exception as e:
                results["fid_error"] = str(e)
                fid_val = torch.tensor([float("nan")], device=device, dtype=torch.float64)
        barrier()
        broadcast(fid_val, src=0)
        results["fid"] = float(fid_val.item())

    if rank == 0:
        order = ["psnr", "ssim", "ms_ssim", "lpips", "dists",
                 "musiq", "brisque", "maniqa", "clipiqa", "niqe", "piqe", "fid"]
        print("\n==== Results (engine_mar-compatible via pyiqa; direct folders) ====")
        for k in order:
            if k in results:
                v = results[k]
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    print(f"{k:8s}: {v}")
                else:
                    print(f"{k:8s}: {v:.6f}")

        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "hr_dir": str(hr_dir),
                "sr_dir": str(sr_dir),
                "matched_pairs_total": len(pairs),
                "align": args.align,
                "device": str(device),
                "world_size": world_size,
                "fid_mode": "direct_input_folders" if not args.no_fid else "disabled",
                "results": results,
            }
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nSaved JSON to: {out_path}")


if __name__ == "__main__":
    main()
