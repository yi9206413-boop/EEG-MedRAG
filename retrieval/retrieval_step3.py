import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 读取问题与超边嵌入
with open(questions_json_path, "r", encoding="utf-8") as f:
    questions = json.load(f)
with open(entity_vdb_path, "r", encoding="utf-8") as f:
    entitys = json.load(f)

# 提取超边 embedding 和 hash
entity_embeddings = np.array([h['embedding'] for h in entitys])
entity_hashes = [h['entity_hash'] for h in entitys]

# 每个问题查找最相似的一个超边（top-1）
top1_hashes = []
for q in tqdm(questions, desc="Embedding & searching top-1 entitys"):
    metadata_adr = q.get("metadata", "")[0]
    with open(metadata_adr) as f:
        metadata = str(json.load(f))
    q_emb = model.encode(metadata, normalize_embeddings=True)
    sims = entity_embeddings @ q_emb
    top1_idx = np.argmax(sims)
    top1_hash = entity_hashes[top1_idx]
    top1_hashes.append(top1_hash)
    
# 加载 hash -> entity_str 映射
with open(entity_list_path, "r", encoding="utf-8") as f:
    entity_list = json.load(f)
hash2hyper = {item["entity_hash"]: item["hyperedge_hash_list"][0] for item in entity_list}
top1_hashes_hyper = [hash2hyper[item] for item in top1_hashes]

# 加载 hash -> entity_str 映射
with open(hyperedge_list_path, "r", encoding="utf-8") as f:
    hyperedge_list = json.load(f)
hash2str = {item["hyperedge_hash"]: item["hyperedge_str"] for item in hyperedge_list}

# 加载 re_kn.json 数据
with open(re_kn_path, "r", encoding="utf-8") as f:
    re_kn_data = json.load(f)

# 插入对应的超边字符串
for i, (item, hash_val) in enumerate(zip(re_kn_data, top1_hashes_hyper)):
    hyperedge_str = hash2str.get(hash_val, "")
    item["entity_retrieval"] = hyperedge_str
    if not hyperedge_str:
        print(f"[警告] 第{i}条未匹配到 entity_str, hash={hash_val}")


import os
import json
from multiprocessing import Pool, cpu_count
from hypergraphrag import HyperGraphRAG, QueryParam
os.environ["OPENAI_API_KEY"] = "sk-Z5r1u2Z8Ukz8Ez81F78f4028B4Bc49548bAa22E4E4FaE19e"
rag = HyperGraphRAG(working_dir="expr/concept_layer")

# # 加载 hash -> entity_str 映射
# with open(entity_list_path, "r", encoding="utf-8") as f:
#     entity_list = json.load(f)
# hash2str = {item["entity_hash"]: item["entity_str"] for item in entity_list}
# top1_strs = [str(item["EEG_retrieval"]) for item in re_kn_data]

top1_strs = []
for d in re_kn_data:
    metadata_adr = d.get("metadata", "")[0]
    with open(metadata_adr) as f:
        metadata = str(json.load(f))
    top1_strs.append(metadata)

# 并行查询函数
def query_knowledge(entity_str):
    result = rag.query(entity_str, QueryParam(only_need_context=True, top_k=2))
    return result

num_processes = min(4, cpu_count())  # 最多用4核
print(f"使用 {num_processes} 个进程进行并行查询...")

with Pool(processes=num_processes) as pool:
    knowledge_results = list(tqdm(pool.imap(query_knowledge, top1_strs), total=len(top1_strs), desc="查询知识"))

# 插入知识到原始数据中
for item, knowledge in zip(re_kn_data, knowledge_results):
    item["knowledge"] = knowledge

# 保存最终结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(re_kn_data, f, indent=2, ensure_ascii=False)

print(f"处理完成，结果已保存至：{output_path}")
