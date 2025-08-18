import os
import json
from hypergraphrag import HyperGraphRAG
os.environ["OPENAI_API_KEY"] = "sk-Z5r1u2Z8Ukz8Ez81F78f4028B4Bc49548bAa22E4E4FaE19e"

rag = HyperGraphRAG(working_dir=f"/home/luohaoran/wy/expr/concept_layer")

with open(f"/home/luohaoran/wy/datasets/knowledge.json", mode="r") as f:
    unique_contexts = json.load(f)
    
rag.insert(unique_contexts)