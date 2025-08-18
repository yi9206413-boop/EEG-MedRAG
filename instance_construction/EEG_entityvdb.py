import json
import os
from tqdm import tqdm
import mne
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


NUM_WORKERS = 60
N_PAA_PIECES = 20  # PAA分段数

def paa(sequence, n_pieces):
    L = len(sequence)
    if L < n_pieces:
        sequence = np.pad(sequence, (0, n_pieces - L))
        L = len(sequence)
    return [float(np.mean(sequence[int(i*L/n_pieces):int((i+1)*L/n_pieces)])) for i in range(n_pieces)]

def process_folder(args):
    hyperedge_hash, address, edf_files = args
    folder_vector = []
    for edf_file in edf_files:
        edf_path = os.path.join(address, edf_file)
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            data = raw.get_data()
            # 每个通道paa降维，然后拼到folder_vector里
            for ch_idx in range(data.shape[0]):
                ch_data = data[ch_idx]
                ch_paa = paa(ch_data, N_PAA_PIECES)
                folder_vector.extend(ch_paa)
        except Exception as e:
            print(f"  Error reading {edf_path}: {e}")
            continue  # 跳过出错的文件
    return {
        "hyperedge_hash": hyperedge_hash,
        "address": address,
        "edf_files": edf_files,
        "vector": folder_vector  # 所有EDF拼在一起的一维向量
    }

if __name__ == "__main__":
    with open(input_path, "r", encoding="utf-8") as f:
        eeg_entities = json.load(f)

    folder_jobs = []
    for entity in eeg_entities:
        hyperedge_hash = entity.get("hyperedge_hash")
        eeg_info = entity.get("EEG", {})
        raw_address = eeg_info.get("address")
        if os.path.isabs(raw_address):
            address = raw_address
        else:
            address = os.path.join(BASE_DIR, raw_address)

        if not os.path.isdir(address):
            print(f"Warning: address not found: {address}")
            continue
        edf_files = [f for f in os.listdir(address) if f.endswith('.edf')]
        if not edf_files:
            continue
        folder_jobs.append((hyperedge_hash, address, edf_files))

    total_jobs = len(folder_jobs)
    print(f"Total EEG entity (folder) jobs to process: {total_jobs}")

    with open(output_path, "w", encoding="utf-8") as fout, \
            ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_job = {executor.submit(process_folder, job): job for job in folder_jobs}
        for i, future in enumerate(tqdm(as_completed(future_to_job), total=total_jobs, desc="Processing Folders (parallel)")):
            result = future.result()
            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()  # 及时落盘

    print(f"Done! All results saved to {output_path} (jsonl, 1 folder/record per line)")
