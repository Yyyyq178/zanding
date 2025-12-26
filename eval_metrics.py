#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_offline_direct_localweights_v2.py

Fixes the 'UnboundLocalError: local variable torch referenced before assignment' by importing torch
before argparse defaults are constructed.

This script evaluates SR vs HR folders with pyiqa, and prevents HuggingFace Hub downloads for
specific repos by mapping repo_id -> local weight file (safetensors/bin). Patch is applied BEFORE
importing timm/pyiqa.

Usage example (offline):
  pip install torch torchvision pillow tqdm pyiqa safetensors
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  python eval_offline_direct_localweights_v2.py \
    --hr_dir /path/HR --sr_dir /path/SR --device cuda:0 --align resize_sr_to_hr \
    --hf_local_weight timm/vit_base_patch8_224.augreg2_in21k_ft_in1k:/abs/path/model.safetensors \
    --out_json results.json
"""

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch  # safe to import early; does not trigger network

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_METRICS = [
    "psnr", "ssim", "lpips", "dists", "ms_ssim",
    "musiq", "brisque", "maniqa", "clipiqa", "niqe", "piqe",
    "fid",
]


def _parse_repo_mapping(items: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for it in items or []:
        if ":" not in it:
            raise ValueError(f"--hf_local_weight expects 'repo_id:/abs/path', got: {it}")
        repo_id, path = it.split(":", 1)
        repo_id = repo_id.strip()
        path = path.strip()
        if not repo_id:
            raise ValueError(f"Empty repo_id in --hf_local_weight: {it}")
        if not path:
            raise ValueError(f"Empty path in --hf_local_weight: {it}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Local weight file not found: {path} (repo_id={repo_id})")
        m[repo_id] = path
    return m


def patch_hf_hub_download(repo_to_local: Dict[str, str], allow_fallback_network: bool = False) -> None:
    if not repo_to_local:
        return

    from safetensors.torch import load_file
    import huggingface_hub
    import huggingface_hub.file_download as hf_fd

    orig = hf_fd.hf_hub_download

    def _mk_tmp_bin(local_st: str) -> str:
        key = hashlib.sha1(local_st.encode("utf-8")).hexdigest()[:12]
        tmp = f"/tmp/{key}_hf_local.pytorch_model.bin"
        if not os.path.exists(tmp):
            sd = load_file(local_st)  # read-only
            torch.save(sd, tmp)
        return tmp

    def _patched(repo_id, filename=None, *args, **kwargs):
        rid = str(repo_id)
        for target_repo, local_path in repo_to_local.items():
            if target_repo in rid:
                # timm commonly requests model.safetensors or pytorch_model.bin
                if filename is None:
                    return local_path
                if filename.endswith(".safetensors"):
                    return local_path
                if filename == "pytorch_model.bin":
                    if local_path.endswith(".safetensors"):
                        return _mk_tmp_bin(local_path)
                    return local_path
                if allow_fallback_network:
                    return orig(repo_id, filename=filename, *args, **kwargs)
                raise FileNotFoundError(
                    f"Offline local-weight mapping matched repo '{target_repo}', "
                    f"but requested filename='{filename}' is not satisfied locally. "
                    f"Add a mapping or use --allow_network_fallback."
                )
        return orig(repo_id, filename=filename, *args, **kwargs)

    hf_fd.hf_hub_download = _patched
    huggingface_hub.hf_hub_download = _patched


def is_dist_avail_and_initialized():
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()


def get_rank():
    import torch.distributed as dist
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    import torch.distributed as dist
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def barrier():
    import torch.distributed as dist
    if is_dist_avail_and_initialized():
        dist.barrier()


def all_reduce_sum(t):
    import torch.distributed as dist
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def broadcast(t, src=0):
    import torch.distributed as dist
    if is_dist_avail_and_initialized():
        dist.broadcast(t, src=src)
    return t


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


def shard_indices(n: int, rank: int, world_size: int) -> List[int]:
    return list(range(rank, n, world_size))


def init_metric_safe(pyiqa_mod, name: str, device):
    try:
        m = pyiqa_mod.create_metric(name, device=device)
        try:
            m.eval()
        except Exception:
            pass
        return m, None
    except Exception as e:
        return None, repr(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--sr_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument(
        "--align",
        type=str,
        default="none",
        choices=["none", "resize_sr_to_hr", "resize_both_to_min", "center_crop_to_min"],
    )
    parser.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--hf_local_weight", action="append", default=[],
        help="Map HF repo_id to local file: repo_id:/abs/path/to/model.safetensors (repeatable)."
    )
    parser.add_argument("--allow_network_fallback", action="store_true")

    # Parse args BEFORE importing timm/pyiqa so patch takes effect.
    args = parser.parse_args()

    repo_to_local = _parse_repo_mapping(args.hf_local_weight)
    patch_hf_hub_download(repo_to_local, allow_fallback_network=args.allow_network_fallback)

    # Heavy imports AFTER patch
    import json
    import math
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.functional as TF

    import pyiqa

    def pil_load_rgb(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def pil_to_tensor_01(img: Image.Image) -> torch.Tensor:
        return TF.to_tensor(img).float()

    class PairedDataset(Dataset):
        def __init__(self, pairs: List[Pair], align: str):
            self.pairs = pairs
            self.align = align

        def __len__(self):
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

    hr_dir = Path(args.hr_dir)
    sr_dir = Path(args.sr_dir)
    if not hr_dir.is_dir():
        raise FileNotFoundError(f"HR dir not found: {hr_dir}")
    if not sr_dir.is_dir():
        raise FileNotFoundError(f"SR dir not found: {sr_dir}")

    rank = get_rank()
    world_size = get_world_size()

    pairs = match_pairs(hr_dir, sr_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if rank == 0:
        print(f"Matched pairs: {len(pairs)}")
        if len(pairs) == 0:
            raise RuntimeError("No matched pairs found (match by relative path without extension).")
        if repo_to_local:
            print("Local HF weight mappings:")
            for k, v in repo_to_local.items():
                print(f"  {k} -> {v}")
    barrier()

    shard = [pairs[i] for i in shard_indices(len(pairs), rank, world_size)]
    dl = DataLoader(
        PairedDataset(shard, align=args.align),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in str(args.device)),
        drop_last=False,
    )

    device = torch.device(args.device)

    full_ref = {"psnr", "ssim", "lpips", "dists", "ms_ssim"}
    want_fid = "fid" in args.metrics

    full_ref_metrics: Dict[str, torch.nn.Module] = {}
    no_ref_metrics: Dict[str, torch.nn.Module] = {}
    issues: Dict[str, str] = {}

    for mname in args.metrics:
        if mname == "fid":
            continue
        module, err = init_metric_safe(pyiqa, mname, device)
        if module is None:
            issues[mname] = err or "unknown error"
            if args.strict:
                raise RuntimeError(f"Metric init failed for '{mname}': {issues[mname]}")
            continue
        if mname in full_ref:
            full_ref_metrics[mname] = module
        else:
            no_ref_metrics[mname] = module

    fid_metric = None
    if want_fid:
        fid_metric, err = init_metric_safe(pyiqa, "fid", device)
        if fid_metric is None:
            issues["fid"] = err or "unknown error"
            if args.strict:
                raise RuntimeError(f"Metric init failed for 'fid': {issues['fid']}")
            want_fid = False

    if rank == 0 and issues:
        print("\n[Warning] Some metrics could not be initialized and will be skipped:")
        for k, v in issues.items():
            print(f"  - {k}: {v}")

    sums: Dict[str, float] = {k: 0.0 for k in list(full_ref_metrics.keys()) + list(no_ref_metrics.keys())}
    counts: Dict[str, int] = {k: 0 for k in sums.keys()}

    for batch in tqdm(dl, desc=f"Rank{rank} metrics", disable=(rank != 0)):
        sr = batch["sr"].to(device, non_blocking=True).float().clamp(0, 1)
        hr = batch["hr"].to(device, non_blocking=True).float().clamp(0, 1)

        if full_ref_metrics and (sr.shape[-2:] != hr.shape[-2:]):
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
    for k in sums.keys():
        s = torch.tensor([sums[k]], device=device, dtype=torch.float64)
        c = torch.tensor([counts[k]], device=device, dtype=torch.float64)
        all_reduce_sum(s)
        all_reduce_sum(c)
        results[k] = float(s.item() / c.item()) if float(c.item()) > 0 else None

    if want_fid and fid_metric is not None:
        fid_val = torch.tensor([float("nan")], device=device, dtype=torch.float64)
        if rank == 0:
            try:
                fid_val = torch.tensor([float(fid_metric(str(sr_dir), str(hr_dir)))], device=device, dtype=torch.float64)
            except Exception as e:
                issues["fid_runtime"] = repr(e)
                fid_val = torch.tensor([float("nan")], device=device, dtype=torch.float64)
        barrier()
        broadcast(fid_val, src=0)
        results["fid"] = float(fid_val.item())

    if rank == 0:
        print("\n==== Results (pyiqa; local-weights override) ====")
        for k in sorted(results.keys()):
            v = results[k]
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                print(f"{k:10s}: {v}")
            else:
                print(f"{k:10s}: {v:.6f}")

        if issues:
            print("\n==== Metric init/runtime issues ====")
            for k, v in issues.items():
                print(f"{k}: {v}")

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
                "requested_metrics": args.metrics,
                "computed_metrics": sorted(list(results.keys())),
                "hf_local_weight": repo_to_local,
                "allow_network_fallback": bool(args.allow_network_fallback),
                "issues": issues,
                "results": results,
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved JSON to: {out_path}")


if __name__ == "__main__":
    main()
