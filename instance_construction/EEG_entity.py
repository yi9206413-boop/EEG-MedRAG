import json



result = []

with open(input_path, "r", encoding="utf-8") as fin:
    data = json.load(fin)

for item in data:
    eeg = item.get("origin", {}).get("EEG")
    if eeg:  # 只要origin里有EEG字段就保留
        result.append({
            "hyperedge_hash": item.get("hyperedge_hash"),
            "EEG": eeg
        })

with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(result, fout, ensure_ascii=False, indent=2)
