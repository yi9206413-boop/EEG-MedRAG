import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



# 加载embedding模型
model = SentenceTransformer('all-MiniLM-L6-v2')  # 如果想用bge-large-zh-v1.5可改这里

# 读取输入文件
with open(input_path, "r", encoding="utf-8") as f:
    hyperedge_list = json.load(f)

# 存放结果
vdb_list = []

for item in tqdm(hyperedge_list, desc="Processing"):
    # 提取hash和str
    hyperedge_hash = item.get("hyperedge_hash", "")
    hyperedge_str = item.get("hyperedge_str", "")
    # 拼接（可以直接embed hyperedge_str，hash可作为id存）
    embedding_input = hyperedge_hash + " " + hyperedge_str

    # 得到embedding向量
    embedding = model.encode(embedding_input, normalize_embeddings=True)
    embedding = embedding.tolist()  # 转为list便于json序列化

    # 构造vdb对象
    vdb_list.append({
        "hyperedge_hash": hyperedge_hash,
        "embedding": embedding,
        "hyperedge_str": hyperedge_str
    })

# 写入输出文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(vdb_list, f, ensure_ascii=False, indent=2)

print(f"Embedding Done! {len(vdb_list)} items saved to {output_path}")
