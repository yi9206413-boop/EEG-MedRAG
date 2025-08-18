#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pyedflib
from pyts.approximation import PiecewiseAggregateApproximation
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq
from typing import List, Dict, Any, Tuple, Optional



PAA_SIZE         = 20
USE_STRUCTURED   = False
DISTANCE_METRIC  = "dtw"
DTW_RADIUS       = 1
NUM_WORKERS      = 60
TOPK             = 1
# ===================================================

def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in open(path) if line.strip()]

def collect_edf_files(eeg_paths: List[str]) -> List[str]:
    edf_files = []
    for p in eeg_paths:
        if not p:
            continue
        if os.path.isfile(p) and p.lower().endswith(".edf"):
            edf_files.append(p)
        elif os.path.isdir(p):
            edf_files.extend(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".edf"))
    return sorted(set(edf_files))

def build_query_vector(edf_files: List[str], paa_size: int, use_structured: bool = False) -> np.ndarray:
    if not edf_files:
        raise ValueError("没有找到任何 EDF 文件，无法构建查询向量")
    with pyedflib.EdfReader(edf_files[0]) as f:
        channel_labels = f.getSignalLabels()
    n_channels = len(channel_labels)
    all_channel_features = [[] for _ in range(n_channels)]
    paa = PiecewiseAggregateApproximation(window_size=None, output_size=paa_size)
    for edf_path in edf_files:
        with pyedflib.EdfReader(edf_path) as reader:
            for ch in range(n_channels):
                sig = reader.readSignal(ch)
                feat = paa.transform(sig.reshape(1, -1)).flatten()
                all_channel_features[ch].append(feat)
    if use_structured:
        per_channel = [np.stack(lst, axis=0) for lst in all_channel_features]
        arr = np.stack(per_channel, axis=0).transpose(1, 0, 2)
        return arr
    else:
        return np.concatenate([np.concatenate(lst, axis=0) for lst in all_channel_features], axis=0)

QUERY_VECTOR: Optional[np.ndarray] = None
DIST_METRIC: str = DISTANCE_METRIC
DTW_R: int = DTW_RADIUS

def _init_worker(query_vec: np.ndarray, dist_metric: str, dtw_radius: int):
    global QUERY_VECTOR, DIST_METRIC, DTW_R
    QUERY_VECTOR = query_vec
    DIST_METRIC = dist_metric
    DTW_R = dtw_radius

def distance_one(item: Dict[str, Any]) -> Tuple[str, float]:
    db_vec = np.asarray(item['vector'], dtype=float)
    if DIST_METRIC == "l2":
        if QUERY_VECTOR.shape != db_vec.shape:
            return item['hyperedge_hash'], float("inf")
        return item['hyperedge_hash'], float(np.sum((QUERY_VECTOR - db_vec) ** 2))
    else:
        dist, _ = fastdtw(QUERY_VECTOR.tolist(), db_vec.tolist(), radius=DTW_R)
        return item['hyperedge_hash'], dist

def load_db_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def load_hyperedge_mapping(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        return {
            entry["hyperedge_hash"]: entry["hyperedge_str"]
            for entry in json.load(f)
        }

def compute_topk(query_vec: np.ndarray, db_items: List[Dict[str, Any]],
                 num_workers: int, metric: str, dtw_radius: int,
                 topk: int) -> List[str]:
    heap: List[Tuple[float, str]] = []
    with ProcessPoolExecutor(max_workers=num_workers,
                             initializer=_init_worker,
                             initargs=(query_vec, metric, dtw_radius)) as ex:
        futures = [ex.submit(distance_one, it) for it in db_items]
        for fut in as_completed(futures):
            try:
                h, d = fut.result()
            except Exception:
                continue
            if len(heap) < topk:
                heapq.heappush(heap, (-d, h))
            else:
                if -heap[0][0] > d:
                    heapq.heapreplace(heap, (-d, h))
    return [h for _, h in sorted([(-d, h) for d, h in heap])]

def main():
    print(f"[Info] 加载问题：{QUESTIONS_PATH}")
    questions = load_questions(QUESTIONS_PATH)
    print(f"[Info] 加载数据库：{DB_JSONL_PATH}")
    db_items = load_db_items(DB_JSONL_PATH)
    print(f"[Info] 加载哈希映射：{HYPEREDGE_PATH}")
    hash_to_str = load_hyperedge_mapping(HYPEREDGE_PATH)

    vector_cache: Dict[str, np.ndarray] = {}
    retrieval_cache: Dict[str, str] = {}  # EEG key -> hyperedge_str
    final_questions = []

    for idx, qobj in enumerate(questions, 1):
        eeg_paths = qobj.get("EEG", [])
        eeg_key = "|".join(sorted(eeg_paths))

        print(f"\n[Q{idx}] {qobj.get('question', '<NO_QUESTION>')}")
        print(f"[Q{idx}] EEG Key: {eeg_key}")

        try:
            if eeg_key in retrieval_cache:
                qobj["EEG_retrieval"] = retrieval_cache[eeg_key]
                print(f"[Q{idx}] 命中缓存: {retrieval_cache[eeg_key]}")
                final_questions.append(qobj)
                continue

            edf_files = collect_edf_files(eeg_paths)
            if not edf_files:
                print(f"[Q{idx}] 无 EDF 文件")
                qobj["EEG_retrieval"] = ""
                final_questions.append(qobj)
                continue

            if eeg_key not in vector_cache:
                vec = build_query_vector(edf_files, PAA_SIZE, USE_STRUCTURED)
                if USE_STRUCTURED:
                    vec = vec.flatten()
                vector_cache[eeg_key] = vec
            else:
                vec = vector_cache[eeg_key]

            top1 = compute_topk(vec, db_items, NUM_WORKERS, DISTANCE_METRIC, DTW_RADIUS, TOPK)
            best_hash = top1[0] if top1 else ""
            hyperedge_str = hash_to_str.get(best_hash, "")

            retrieval_cache[eeg_key] = hyperedge_str
            qobj["EEG_retrieval"] = hyperedge_str
            print(f"[Q{idx}] 计算并缓存: {hyperedge_str}")

        except Exception as e:
            print(f"[Q{idx}] 错误: {e}")
            qobj["EEG_retrieval"] = ""

        final_questions.append(qobj)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_questions, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] 已写入结果至：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
