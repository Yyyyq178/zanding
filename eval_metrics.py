#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_metrics.py

修复日志 (FIX LOG):
- [CRITICAL] 修复 'ValueError: This file contains pickled data'。
- 解决方法：在 np.load() 中添加 allow_pickle=True。
- 之前的所有修复（AttributeError, 手动计算 FID_FFHQ）均已包含。

Usage:
  python eval_metrics.py \
    --hr_dir /path/HR --sr_dir /path/SR --device cuda:0 \
    --metrics psnr ssim fid fid_ffhq deg lmd \
    --fid_stats pretrained_models/ffhq_stats.npz
"""

import argparse
import hashlib
import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# --- Optional Imports for Face Metrics ---
try:
    import dlib
    from imutils import face_utils
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False

try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

# --- Optional Import for FID (Low-level API) ---
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
    HAS_PYTORCH_FID = True
except ImportError:
    HAS_PYTORCH_FID = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_METRICS = [
    "psnr", "ssim", "lpips", "dists", "ms_ssim",
    "musiq", "brisque", "maniqa", "clipiqa", "niqe", "piqe",
    "fid", "fid_ffhq", "deg", "lmd"
]

FULL_REF_METRIC_NAMES = {"psnr", "ssim", "lpips", "dists", "ms_ssim", "deg", "lmd"}

# --- Face Metrics Calculator Class ---
class FaceMetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.app = None
        self.detector = None
        self.predictor = None
        
        # Initialize InsightFace
        if HAS_INSIGHTFACE:
            providers = ['CUDAExecutionProvider'] if 'cuda' in str(device).lower() else ['CPUExecutionProvider']
            try:
                self.app = FaceAnalysis(name='buffalo_l', providers=providers)
                self.app.prepare(ctx_id=0, det_size=(512, 512))
            except Exception as e:
                print(f"[FaceMetrics] Failed to init InsightFace: {e}")
                self.app = None
        else:
            print("[FaceMetrics] InsightFace not installed. 'deg' metric will be unavailable.")

        # Initialize Dlib
        if HAS_DLIB:
            self.detector = dlib.get_frontal_face_detector()
            model_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(model_path):
                try:
                    self.predictor = dlib.shape_predictor(model_path)
                except Exception as e:
                    print(f"[FaceMetrics] Failed to load dlib predictor: {e}")
                    self.predictor = None
            else:
                print(f"[FaceMetrics] '{model_path}' not found. 'lmd' metric will be unavailable.")
        else:
            print("[FaceMetrics] dlib not installed. 'lmd' metric will be unavailable.")

    def _tensor_to_numpy_image(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = img[:, :, ::-1] # RGB to BGR
        return img

    def compute_batch(self, sr_batch, hr_batch):
        deg_scores = []
        lmd_scores = []
        batch_size = sr_batch.shape[0]
        
        for i in range(batch_size):
            sr_img = self._tensor_to_numpy_image(sr_batch[i])
            hr_img = self._tensor_to_numpy_image(hr_batch[i])

            # Deg
            val_deg = np.nan
            if self.app is not None:
                try:
                    faces_sr = self.app.get(sr_img)
                    faces_hr = self.app.get(hr_img)
                    if len(faces_sr) > 0 and len(faces_hr) > 0:
                        idx_sr = np.argmax([f.det_score for f in faces_sr])
                        idx_hr = np.argmax([f.det_score for f in faces_hr])
                        emb_sr = faces_sr[idx_sr].embedding
                        emb_hr = faces_hr[idx_hr].embedding
                        norm_sr = np.linalg.norm(emb_sr)
                        norm_hr = np.linalg.norm(emb_hr)
                        cosine = np.dot(emb_sr, emb_hr) / (norm_sr * norm_hr + 1e-6)
                        cosine = np.clip(cosine, -1.0, 1.0)
                        val_deg = np.degrees(np.arccos(cosine))
                except Exception:
                    pass
            deg_scores.append(val_deg)

            # LMD
            val_lmd = np.nan
            if self.detector is not None and self.predictor is not None:
                try:
                    rects_sr = self.detector(sr_img, 1)
                    rects_hr = self.detector(hr_img, 1)
                    if len(rects_sr) > 0 and len(rects_hr) > 0:
                        shape_sr = self.predictor(sr_img, rects_sr[0])
                        shape_hr = self.predictor(hr_img, rects_hr[0])
                        pts_sr = face_utils.shape_to_np(shape_sr)
                        pts_hr = face_utils.shape_to_np(shape_hr)
                        val_lmd = np.linalg.norm(pts_sr - pts_hr, axis=1).mean()
                except Exception:
                    pass
            lmd_scores.append(val_lmd)

        return deg_scores, lmd_scores

# --- Helper Functions ---

def _parse_repo_mapping(items: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for it in items or []:
        if ":" not in it: raise ValueError(f"Invalid mapping: {it}")
        repo_id, path = it.split(":", 1)
        m[repo_id.strip()] = path.strip()
    return m

def patch_hf_hub_download(repo_to_local: Dict[str, str], allow_fallback_network: bool = False) -> None:
    if not repo_to_local: return
    from safetensors.torch import load_file
    import huggingface_hub
    import huggingface_hub.file_download as hf_fd
    orig = hf_fd.hf_hub_download

    def _mk_tmp_bin(local_st: str) -> str:
        key = hashlib.sha1(local_st.encode("utf-8")).hexdigest()[:12]
        tmp = f"/tmp/{key}_hf_local.pytorch_model.bin"
        if not os.path.exists(tmp):
            sd = load_file(local_st)
            torch.save(sd, tmp)
        return tmp

    def _patched(repo_id, filename=None, *args, **kwargs):
        rid = str(repo_id)
        for target_repo, local_path in repo_to_local.items():
            if target_repo in rid:
                if filename is None: return local_path
                if filename.endswith(".safetensors"): return local_path
                if filename == "pytorch_model.bin":
                    if local_path.endswith(".safetensors"): return _mk_tmp_bin(local_path)
                    return local_path
                if allow_fallback_network: return orig(repo_id, filename=filename, *args, **kwargs)
                raise FileNotFoundError(f"Offline mapping failed for {repo_id}/{filename}")
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
    if is_dist_avail_and_initialized(): dist.barrier()

def all_reduce_sum(t):
    import torch.distributed as dist
    if is_dist_avail_and_initialized(): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def broadcast(t, src=0):
    import torch.distributed as dist
    if is_dist_avail_and_initialized(): dist.broadcast(t, src=src)
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
        if key not in m: m[key] = p
    return m

@dataclass
class Pair:
    key: str
    sr: Path
    hr: Optional[Path] = None

def match_pairs(hr_dir: Optional[Path], sr_dir: Path) -> List[Pair]:
    sr_map = _build_key_map(sr_dir)
    if hr_dir is None:
        keys = sorted(sr_map.keys())
        return [Pair(k, sr_map[k], None) for k in keys]
    else:
        hr_map = _build_key_map(hr_dir)
        keys = sorted(set(hr_map.keys()).intersection(sr_map.keys()))
        return [Pair(k, sr_map[k], hr_map[k]) for k in keys]

def shard_indices(n: int, rank: int, world_size: int) -> List[int]:
    return list(range(rank, n, world_size))

def init_metric_safe(pyiqa_mod, name: str, device):
    try:
        m = pyiqa_mod.create_metric(name, device=device)
        try: m.eval()
        except Exception: pass
        return m, None
    except Exception as e:
        return None, repr(e)

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, default=None, help="Optional HR dir")
    parser.add_argument("--sr_dir", type=str, required=True)
    parser.add_argument("--fid_stats", type=str, default="pretrained_models/ffhq_stats.npz", 
                        help="Path to pre-calculated FID statistics (.npz file).")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--align", type=str, default="none", choices=["none", "resize_sr_to_hr", "resize_both_to_min", "center_crop_to_min"])
    parser.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--hf_local_weight", action="append", default=[])
    parser.add_argument("--allow_network_fallback", action="store_true")

    args = parser.parse_args()
    repo_to_local = _parse_repo_mapping(args.hf_local_weight)
    patch_hf_hub_download(repo_to_local, allow_fallback_network=args.allow_network_fallback)

    import json
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
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx: int):
            p = self.pairs[idx]
            sr_img = pil_load_rgb(p.sr)
            hr_img = None
            if p.hr is not None:
                hr_img = pil_load_rgb(p.hr)
                if self.align == "resize_sr_to_hr":
                    if sr_img.size != hr_img.size: sr_img = sr_img.resize(hr_img.size, resample=Image.BICUBIC)
                elif self.align == "resize_both_to_min":
                    if sr_img.size != hr_img.size:
                        w, h = min(sr_img.size[0], hr_img.size[0]), min(sr_img.size[1], hr_img.size[1])
                        sr_img = sr_img.resize((w, h), resample=Image.BICUBIC)
                        hr_img = hr_img.resize((w, h), resample=Image.BICUBIC)
                elif self.align == "center_crop_to_min":
                    if sr_img.size != hr_img.size:
                        w, h = min(sr_img.size[0], hr_img.size[0]), min(sr_img.size[1], hr_img.size[1])
                        sr_img = TF.center_crop(sr_img, [h, w])
                        hr_img = TF.center_crop(hr_img, [h, w])
            sr = pil_to_tensor_01(sr_img).clamp(0, 1)
            hr = pil_to_tensor_01(hr_img).clamp(0, 1) if hr_img is not None else torch.zeros(1)
            return {"key": p.key, "hr": hr, "sr": sr}

    # Data Checks
    sr_dir = Path(args.sr_dir)
    if not sr_dir.is_dir(): raise FileNotFoundError(f"SR dir not found: {sr_dir}")
    hr_dir = Path(args.hr_dir) if args.hr_dir else None
    if hr_dir and not hr_dir.is_dir(): raise FileNotFoundError(f"HR dir not found: {hr_dir}")

    rank = get_rank()
    world_size = get_world_size()
    
    # 1. 匹配
    pairs = match_pairs(hr_dir, sr_dir)
    if args.limit > 0: pairs = pairs[: args.limit]

    if rank == 0:
        print(f"Matched pairs: {len(pairs)}")
        if hr_dir is None: print("Notice: No HR provided. Skipping Full-Ref metrics.")
    barrier()

    # 2. 切片 (Fix: 之前这里漏了)
    indices = shard_indices(len(pairs), rank, world_size)
    shard = [pairs[i] for i in indices]

    # 3. Loader
    dl = DataLoader(
        PairedDataset(shard, align=args.align),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=("cuda" in str(args.device)), drop_last=False
    )

    device = torch.device(args.device)
    
    # Flags
    want_fid = "fid" in args.metrics
    want_fid_ffhq = "fid_ffhq" in args.metrics
    want_deg = "deg" in args.metrics
    want_lmd = "lmd" in args.metrics

    if hr_dir is None:
        if want_fid: want_fid = False
        if want_deg: want_deg = False
        if want_lmd: want_lmd = False
        args.metrics = [m for m in args.metrics if m not in FULL_REF_METRIC_NAMES and m != "fid"]

    # Init pyiqa metrics
    full_ref_metrics = {}
    no_ref_metrics = {}
    issues = {}

    for mname in args.metrics:
        if mname in ["fid", "fid_ffhq", "deg", "lmd"]: continue
        dev = torch.device('cpu') if mname == 'niqe' else device
        mod, err = init_metric_safe(pyiqa, mname, dev)
        if mod is None:
            issues[mname] = err
            continue
        if mname in FULL_REF_METRIC_NAMES: full_ref_metrics[mname] = mod
        else: no_ref_metrics[mname] = mod

    # Init Standard FID (vs HR)
    fid_metric = None
    if want_fid:
        fid_metric, err = init_metric_safe(pyiqa, "fid", device)
        if fid_metric is None:
            issues["fid"] = err
            want_fid = False

    # Init Face metrics
    face_calculator = None
    if (want_deg or want_lmd) and hr_dir is not None:
        face_calculator = FaceMetricsCalculator(device)

    # Accumulators
    all_keys = list(full_ref_metrics.keys()) + list(no_ref_metrics.keys())
    if want_deg: all_keys.append("deg")
    if want_lmd: all_keys.append("lmd")
    sums = {k: 0.0 for k in all_keys}
    counts = {k: 0 for k in all_keys}

    # Main Evaluation Loop
    for batch in tqdm(dl, desc=f"Rank{rank} metrics", disable=(rank != 0)):
        sr = batch["sr"].to(device, non_blocking=True).float().clamp(0, 1)
        hr = batch["hr"].to(device, non_blocking=True).float().clamp(0, 1) if hr_dir else None

        with torch.no_grad():
            if hr is not None and full_ref_metrics:
                for n, m in full_ref_metrics.items():
                    v = m(sr, hr).mean()
                    sums[n] += float(v.cpu()) * sr.size(0)
                    counts[n] += sr.size(0)
            for n, m in no_ref_metrics.items():
                v = m(sr).mean()
                sums[n] += float(v.cpu()) * sr.size(0)
                counts[n] += sr.size(0)
            if face_calculator and hr is not None:
                deg_v, lmd_v = face_calculator.compute_batch(sr, hr)
                if want_deg:
                    valid = [x for x in deg_v if not np.isnan(x)]
                    if valid:
                        sums["deg"] += sum(valid)
                        counts["deg"] += len(valid)
                if want_lmd:
                    valid = [x for x in lmd_v if not np.isnan(x)]
                    if valid:
                        sums["lmd"] += sum(valid)
                        counts["lmd"] += len(valid)

    # Reduce Standard Metrics
    results = {}
    for k in sums.keys():
        s = torch.tensor([sums[k]], device=device, dtype=torch.float64)
        c = torch.tensor([counts[k]], device=device, dtype=torch.float64)
        all_reduce_sum(s)
        all_reduce_sum(c)
        results[k] = float(s.item()/c.item()) if c.item() > 0 else None

    # --- FID (Standard: vs HR) ---
    if want_fid and fid_metric and hr_dir:
        val = torch.tensor([float("nan")], device=device, dtype=torch.float64)
        if rank == 0:
            try:
                val = torch.tensor([float(fid_metric(str(sr_dir), str(hr_dir)))], device=device)
            except Exception as e:
                issues["fid_runtime"] = repr(e)
        barrier()
        broadcast(val, src=0)
        results["fid"] = float(val.item())

    # --- FID_FFHQ (Manual robust) ---
    if want_fid_ffhq:
        val_ffhq = torch.tensor([float("nan")], device=device, dtype=torch.float64)
        if rank == 0:
            if not HAS_PYTORCH_FID:
                print("[Error] pytorch-fid not installed.")
                issues["fid_ffhq"] = "pytorch-fid missing"
            elif not os.path.exists(args.fid_stats):
                print(f"[Error] Stats file missing: {args.fid_stats}")
                issues["fid_ffhq"] = "file missing"
            else:
                try:
                    print(f"Calculating FID_FFHQ against: {args.fid_stats} ...")
                    
                    # 1. Load Stats Manually (Fix for pickle issue)
                    # allow_pickle=True is needed for .npz saved with object arrays
                    stats = np.load(args.fid_stats, allow_pickle=True)
                    mu2 = stats['mu']
                    sigma2 = stats['sigma']
                    
                    # 2. Init Inception
                    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                    model = InceptionV3([block_idx]).to(device)
                    
                    # 3. Compute SR Stats
                    mu1, sigma1 = compute_statistics_of_path(
                        str(sr_dir), 
                        model, 
                        batch_size=args.batch_size, 
                        dims=2048, 
                        device=device,
                        num_workers=args.num_workers
                    )
                    
                    # 4. Calculate Distance
                    fid_val = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
                    
                    val_ffhq = torch.tensor([float(fid_val)], device=device)
                    print(f"FID_FFHQ Success: {fid_val}")
                    
                except Exception as e:
                    print(f"FID_FFHQ failed: {e}")
                    import traceback
                    traceback.print_exc()
                    issues["fid_ffhq_runtime"] = repr(e)

        barrier()
        broadcast(val_ffhq, src=0)
        results["fid_ffhq"] = float(val_ffhq.item())

    # Output
    if rank == 0:
        print("\n==== Results ====")
        for k in sorted(results.keys()):
            v = results[k]
            print(f"{k:10s}: {v if v is None else f'{v:.6f}'}")
        if issues:
            print("\n==== Issues ====")
            for k, v in issues.items(): print(f"{k}: {v}")
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "hr_dir": str(hr_dir) if hr_dir else None,
                "sr_dir": str(sr_dir),
                "metrics": results
            }
            Path(args.out_json).write_text(json.dumps(payload, indent=2))
            print(f"\nSaved JSON to: {args.out_json}")

if __name__ == "__main__":
    main()