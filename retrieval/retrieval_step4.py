#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pyedflib
import traceback
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm



TARGET_FS = 256            # 统一采样率 (下采样目标)
DOWNSAMPLE_WHEN_HIGHER = True
UPSAMPLE_WHEN_LOWER    = False   # 低于 TARGET_FS 是否上采样
PAA_PER_CHANNEL_SEG    = 8       # 每通道 PAA 段数
FINAL_DIM              = 100     # 固定输出维度
NORM_MODE              = "zscore_channel"  # "zscore_channel" | "minmax_channel" | "none" | "zscore_global"
USE_FAST_PAA           = True
MAX_WORKERS            = 40       # 并行进程数
CHUNK_WRITE            = False    # 如需分块增量写，可置 True

SEARCH_KEYS = [
    ("EEG", "EEG_features"),
    ("EEG_retrieval", "EEG_retrieval_features"),
    ("hyperedge_retrieval", "hyperedge_retrieval_features"),
    ("entity_retrieval", "entity_retrieval_features"),
]
# ==========================================

BASE_DIR = os.path.dirname(JSON_IN)

# ----- 尝试 scipy 优化重采样 -----
try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    print("[WARN] 未检测到 scipy，将使用线性插值重采样。可安装: pip install scipy")


# ================== PAA 实现 ==================
def paa_round(sig: np.ndarray, seg: int) -> np.ndarray:
    n = len(sig)
    if seg <= 0:
        raise ValueError("seg 必须 > 0")
    if n < seg:
        sig = np.pad(sig, (0, seg - n))
        n = len(sig)
    if n % seg == 0:
        return sig.reshape(seg, -1).mean(axis=1)
    out = np.zeros(seg)
    for i in range(seg):
        a = int(round(i * n / seg))
        b = int(round((i + 1) * n / seg))
        if b <= a:
            b = min(a + 1, n)
        out[i] = sig[a:b].mean()
    return out

def paa_floor(sig: np.ndarray, seg: int) -> np.ndarray:
    if seg <= 0:
        raise ValueError("seg 必须 > 0")
    n = len(sig)
    if n < seg:
        sig = np.pad(sig, (0, seg - n))
        n = len(sig)
    idx = (np.floor(np.arange(n) * seg / n)).astype(int)
    idx[idx >= seg] = seg - 1
    sums = np.bincount(idx, weights=sig, minlength=seg)
    counts = np.bincount(idx, minlength=seg)
    return sums / (counts + 1e-12)

def paa(sig: np.ndarray, seg: int) -> np.ndarray:
    return paa_floor(sig, seg) if USE_FAST_PAA else paa_round(sig, seg)


# ================== 工具函数 ==================
def normalize_path(path: str) -> str:
    if not path:
        return path
    path = path.strip().strip('"').strip()
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(BASE_DIR, path.lstrip("/")))

def first_edf_file(path: str) -> str | None:
    if not path:
        return None
    path = normalize_path(path)
    if os.path.isfile(path) and path.lower().endswith(".edf"):
        return path
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            edfs = sorted([f for f in files if f.lower().endswith(".edf")])
            if edfs:
                return os.path.join(root, edfs[0])
    return None

def resample_signal(sig: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    if orig_fs == target_fs:
        return sig
    if orig_fs > target_fs and DOWNSAMPLE_WHEN_HIGHER:
        ratio = target_fs / orig_fs
    elif orig_fs < target_fs and UPSAMPLE_WHEN_LOWER:
        ratio = target_fs / orig_fs
    else:
        return sig

    if _HAS_SCIPY:
        from math import gcd
        scale = 10000
        up = int(round(ratio * scale))
        down = scale
        g = gcd(up, down)
        up //= g
        down //= g
        return resample_poly(sig, up, down)
    else:
        n = len(sig)
        duration = n / orig_fs
        new_len = int(round(duration * target_fs))
        if new_len <= 1:
            return sig
        t_old = np.linspace(0, duration, n, endpoint=False)
        t_new = np.linspace(0, duration, new_len, endpoint=False)
        return np.interp(t_new, t_old, sig)

def interpolate_to_length(vec: np.ndarray, final_len: int) -> np.ndarray:
    L = len(vec)
    if final_len is None:
        return vec
    if L == final_len:
        return vec
    if L <= 1:
        return np.pad(vec, (0, max(0, final_len - L)))[:final_len]
    x_old = np.linspace(0, 1, L)
    x_new = np.linspace(0, 1, final_len)
    return np.interp(x_new, x_old, vec)

# ================== 归一化 ==================
def normalize_channels(all_channels: list[np.ndarray], mode: str) -> list[np.ndarray]:
    if mode == "none":
        return all_channels
    if mode == "zscore_channel":
        out = []
        for sig in all_channels:
            mu = sig.mean()
            sigma = sig.std()
            out.append((sig - mu) / (sigma + 1e-8))
        return out
    if mode == "minmax_channel":
        out = []
        for sig in all_channels:
            mn = sig.min()
            mx = sig.max()
            if mx - mn < 1e-12:
                out.append(np.zeros_like(sig))
            else:
                out.append((sig - mn) / (mx - mn))
        return out
    if mode == "zscore_global":
        concat = np.concatenate(all_channels)
        mu = concat.mean()
        sigma = concat.std()
        return [(sig - mu) / (sigma + 1e-8) for sig in all_channels]
    raise ValueError(f"未知 NORM_MODE: {mode}")

# ================== EDF -> 100维特征 ==================
def edf_feature_vector_100(edf_path: str) -> list[float]:
    reader = pyedflib.EdfReader(edf_path)
    try:
        n_ch = reader.signals_in_file
        channel_signals = []
        for ch in range(n_ch):
            sig = reader.readSignal(ch)
            fs = reader.getSampleFrequency(ch)
            sig = resample_signal(sig, fs, TARGET_FS)
            channel_signals.append(sig.astype(float))

        channel_signals = normalize_channels(channel_signals, NORM_MODE)
        per_channel_feats = [paa(sig, PAA_PER_CHANNEL_SEG) for sig in channel_signals]
        concat_vec = np.concatenate(per_channel_feats) if per_channel_feats else np.zeros(1)
        final_vec = interpolate_to_length(concat_vec, FINAL_DIM)
        return final_vec.astype(float).tolist()
    finally:
        reader.close()

# ================== 地址解析 ==================
def extract_address(entry: dict, key: str) -> str | None:
    if key not in entry:
        return None
    if key == "EEG":
        v = entry["EEG"]
        if isinstance(v, list) and v:
            return v[0]
        return None
    raw = entry[key]
    if isinstance(raw, str):
        try:
            inner = json.loads(raw)
            return inner.get("EEG", {}).get("address")
        except Exception:
            return None
    return None

# ================== 并行任务函数 ==================
def worker_task(args):
    """
    独立进程执行：读取 EDF 计算 100 维特征
    返回 (sample_index, out_field, vector or None, error_msg)
    """
    sample_index, field, out_field, addr = args
    try:
        edf_path = first_edf_file(addr)
        if not edf_path:
            return (sample_index, out_field, None, f"未找到EDF: {addr}")
        vec = edf_feature_vector_100(edf_path)
        vec = [round(float(x), 4) for x in vec]
        return (sample_index, out_field, vec, None)
    except Exception as e:
        return (sample_index, out_field, None, f"{type(e).__name__}: {e}")

# ================== 安全写 ==================
def atomic_write(path: str, data):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d,
                                     suffix=".tmp", encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    shutil.move(tmp_name, path)

# ================== 主流程 ==================
def main():
    with open(JSON_IN, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        samples = [data]
        was_list = False
    elif isinstance(data, list):
        samples = data
        was_list = True
    else:
        raise TypeError("顶层 JSON 必须是对象或数组。")

    print(f"[INFO] 样本数: {len(samples)} | NORM_MODE={NORM_MODE} | PAA_SEG={PAA_PER_CHANNEL_SEG} | WORKERS={MAX_WORKERS}")

    # 1. 构建任务列表
    tasks = []
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        for field, out_field in SEARCH_KEYS:
            addr = extract_address(sample, field)
            if not addr:
                continue
            tasks.append((i, field, out_field, addr))

    print(f"[INFO] 待处理任务数: {len(tasks)}")

    if not tasks:
        print("[INFO] 无任务，直接写出原数据。")
        atomic_write(JSON_OUT, samples if was_list else samples[0])
        return

    # 2. 并行执行
    results_done = 0
    errors = []

    # tqdm 进度条
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(worker_task, t): t for t in tasks}
        pbar = tqdm(total=len(tasks), desc="Processing", ncols=90)
        for fut in as_completed(future_map):
            sample_index, out_field, vec, err = fut.result()
            if err is None and vec is not None:
                samples[sample_index][out_field] = vec
            else:
                errors.append((sample_index, out_field, err))
            results_done += 1
            pbar.update(1)

            # 可选分块增量写
            if CHUNK_WRITE and results_done % 50 == 0:
                atomic_write(JSON_OUT, samples if was_list else samples[0])
        pbar.close()

    # 3. 汇总错误
    if errors:
        print("[WARN] 发生以下错误：")
        for si, field, emsg in errors[:20]:
            print(f"  - 样本 {si} 字段 {field}: {emsg}")
        if len(errors) > 20:
            print(f"  ... 其余 {len(errors)-20} 条省略")

    # 4. 最终写出
    atomic_write(JSON_OUT, samples if was_list else samples[0])
    print(f"[OK] 写入完成: {JSON_OUT}")

if __name__ == "__main__":
    # 可选：限制底层数学库线程，防止超额占用
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
