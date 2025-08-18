import json
import hashlib



def hyperedge_to_hash(hyperedge):
    hyperedge_str = json.dumps(hyperedge, sort_keys=True, ensure_ascii=False)
    hash_val = hashlib.sha256(hyperedge_str.encode('utf-8')).hexdigest()
    return hash_val, hyperedge_str

# 读取原始json
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

hyperedge_list = []
for patient in data:
    hash_val, hyperedge_str = hyperedge_to_hash(patient)
    hyperedge_list.append({
        'hyperedge_hash': hash_val,
        'hyperedge_str': hyperedge_str,
        'origin': patient  # 原始结构可选，保留方便溯源，也可只留hash和str
    })

# 输出到目标json
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(hyperedge_list, f, indent=2, ensure_ascii=False)

print(f"已输出到 {output_path}")
