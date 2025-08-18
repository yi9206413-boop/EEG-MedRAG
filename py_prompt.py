import json
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  # 自动选择合适的可视化（notebook / 控制台）

# 读取 API 密钥
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

# 初始化客户端
client = OpenAI(base_url="https://vip.apiyi.com/v1")

BASELINE = "ours"    
MODEL = "model name"        

# 读取数据
with open("retrieval_result2.json") as f:
    data = json.load(f)

def process_item(d):
    try:
        
        if BASELINE == "OURS":
            retrieval = {
                "EEG_retrieval": d["EEG_retrieval"],
                "EEG_retrieval_features": d["EEG_retrieval_features"],
                "hyperedge_retrieval": d["hyperedge_retrieval"],
                "hyperedge_retrieval_features": d["hyperedge_retrieval_features"],
                "entity_retrieval": d["entity_retrieval"],
                "entity_retrieval_features": d["entity_retrieval_features"],
                "knowledge": d["knowledge"]
            }

        # 读取 metadata 文件（假设每条只取列表第一个路径）
        try:
            with open(d["metadata"][0]) as f:
                metadata = json.load(f)
        except:
            metadata = str(d["metadata"][0])

        question = {
            "metadata": str(metadata),
            "EEG_features": d["EEG_features"],
            "question": d["question"],
        }
        question_str = str(question)

        prompt = f"""---Role---

You are a helpful assistant responding to questions based on given knowledge.

---Knowledge---

{retrieval_str}

---Goal---

Answer the given question.
You must first conduct reasoning inside <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.
The <think>...</think> should as comprehensive as posssible.
The <answer>...</answer> must be **no more than 20 words**.

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>

---Question---

{question_str}
"""
        d["prompt"] = prompt

        response = client.chat.completions.create(
            model=f"{MODEL}",
            messages=[{"role": "user", "content": prompt}],
        )
        d["generation"] = response.choices[0].message.content
        return d

    except Exception as e:
        d["generation"] = f"Error: {e}"
        return d


# 并发 + 进度条
results = []
with ThreadPoolExecutor(max_workers=70) as executor:
    futures = [executor.submit(process_item, d) for d in data]
    with tqdm(total=len(futures), desc="Generating", unit="item") as pbar:
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)

# 如果需要保持与原 data 相同顺序，可按某个唯一键（如 d['id']）重新排序。
# 这里假设不需要；若需要，请在提交前捕捉索引并返回 (idx, d) 再恢复顺序。

# 保存结果
save_dir = f"test_generation_{BASELINE}_{MODEL}.json"
with open(save_dir, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {save_dir}")
