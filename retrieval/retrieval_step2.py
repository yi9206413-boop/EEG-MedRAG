import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 读取问题与超边嵌入
with open(questions_json_path, "r", encoding="utf-8") as f:
    questions = json.load(f)
with open(hyperedge_vdb_path, "r", encoding="utf-8") as f:
    hyperedges = json.load(f)

# 提取超边 embedding 和 hash
hyperedge_embeddings = np.array([h['embedding'] for h in hyperedges])
hyperedge_hashes = [h['hyperedge_hash'] for h in hyperedges]

# 每个问题查找最相似的一个超边（top-1）
top1_hashes = []
for q in tqdm(questions, desc="Embedding & searching top-1 hyperedges"):
    metadata_adr = q.get("metadata", "")[0]
    with open(metadata_adr) as f:
        metadata = str(json.load(f))
    q_emb = model.encode(metadata, normalize_embeddings=True)
    sims = hyperedge_embeddings @ q_emb
    top1_idx = np.argmax(sims)
    top1_hash = hyperedge_hashes[top1_idx]
    top1_hashes.append(top1_hash)

# 加载 hash -> hyperedge_str 映射
with open(hyperedge_list_path, "r", encoding="utf-8") as f:
    hyperedge_list = json.load(f)
hash2str = {item["hyperedge_hash"]: item["hyperedge_str"] for item in hyperedge_list}

# 加载 re_kn.json 数据
with open(re_kn_path, "r", encoding="utf-8") as f:
    re_kn_data = json.load(f)

# 插入对应的超边字符串
for i, (item, hash_val) in enumerate(zip(re_kn_data, top1_hashes)):
    hyperedge_str = hash2str.get(hash_val, "")
    item["hyperedge_retrieval"] = hyperedge_str
    if not hyperedge_str:
        print(f"[警告] 第{i}条未匹配到 hyperedge_str, hash={hash_val}")

# 保存最终结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(re_kn_data, f, indent=2, ensure_ascii=False)

print(f"处理完成，结果已保存至：{output_path}")
