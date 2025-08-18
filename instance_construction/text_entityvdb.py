import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



# 选用的embedding模型
model = SentenceTransformer('all-MiniLM-L6-v2')  # 如果想用别的模型可以替换这里

# 加载实体列表
with open(input_path, "r", encoding="utf-8") as f:
    entity_list = json.load(f)

vdb_list = []

for item in tqdm(entity_list, desc="Embedding entity_str"):
    entity_str = item.get("entity_str", "")
    entity_hash = item.get("entity_hash", "")
    hyperedge_hash_list = item.get("hyperedge_hash_list", [])

    # 对entity_str做embedding
    embedding = model.encode(entity_str, normalize_embeddings=True)
    embedding = embedding.tolist()

    vdb_list.append({
        "entity_hash": entity_hash,
        "entity_str": entity_str,
        "embedding": embedding,
        "hyperedge_hash_list": hyperedge_hash_list
    })

# 输出到json文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(vdb_list, f, ensure_ascii=False, indent=2)

print(f"Embedding complete! {len(vdb_list)} entities saved to {output_path}")
