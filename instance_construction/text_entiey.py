import json
import hashlib
from collections import defaultdict

# 输入输出路径
input_path = '/home/luohaoran/wy/expr/instance_layer/hyperedge_list.json'
output_path = '/home/luohaoran/wy/expr/instance_layer/text_entity_list.json'

def entity_hash(s):
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()

def extract_text_entities(obj):
    """
    递归提取所有叶子节点（非dict、非list），返回所有实体字符串（set）
    忽略 patient_id 和 EEG 字段（连同EEG的整个内容）
    """
    entities = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'EEG' or k == 'patient_id':
                continue  # 跳过 EEG 整个字段及 patient_id
            entities |= extract_text_entities(v)
    elif isinstance(obj, list):
        for i in obj:
            entities |= extract_text_entities(i)
    else:
        entities.add(str(obj))  # 原子实体
    return entities

# 读入 hyperedge 文件
with open(input_path, 'r', encoding='utf-8') as f:
    hyperedges = json.load(f)

# 构建 entity -> set(hyperedge_hash)
entity_hyperedges = defaultdict(set)

for he in hyperedges:
    hyperedge_hash = he["hyperedge_hash"]
    origin = he["origin"]
    entities = extract_text_entities(origin)
    for e in entities:
        entity_hyperedges[e].add(hyperedge_hash)

# 构建输出
entity_list = []
for ent, he_hash_set in entity_hyperedges.items():
    entity_list.append({
        "entity_str": ent,
        "entity_hash": entity_hash(ent),
        "hyperedge_hash_list": sorted(list(he_hash_set))
    })

# 写入输出文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(entity_list, f, ensure_ascii=False, indent=2)

print(f"已输出到 {output_path}")
