import os
from hypergraphrag import HyperGraphRAG, QueryParam
os.environ["OPENAI_API_KEY"] = " "

rag = HyperGraphRAG(working_dir=f"expr/concept_layer")

query_text = 'MHRA'

result = rag.query(query_text, QueryParam(only_need_context=True,top_k=1))
print(result)
