import os
from hypergraphrag import HyperGraphRAG, QueryParam
os.environ["OPENAI_API_KEY"] = "sk-Z5r1u2Z8Ukz8Ez81F78f4028B4Bc49548bAa22E4E4FaE19e"

rag = HyperGraphRAG(working_dir=f"expr/concept_layer")

query_text = 'MHRA'

result = rag.query(query_text, QueryParam(only_need_context=True,top_k=1))
print(result)