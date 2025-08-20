# EEG-MedRAG

EEG-MedRAG:EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation [[paper](https://arxiv.org/abs/2503.21322)]

##  Overview 

![](./figs/[F3.pdf](https://github.com/user-attachments/files/21887621/F3.pdf)
F1.png)

## Environment Setup
```bash
conda create python=3.11
conda activate ptthon3.11
pip install -r requirements.txt(https://github.com/user-attachments/files/21887616/F3.pdf)

```

Then, we need to set openai api key in openai_api_key.txt file. (We use www.apiyi.com for LLM server.)

Last, we need download the contexts and datasets from https://physionet.org/content/chbmit/1.0.0/ and https://openneuro.org/ and put them in the contexts and datasets folders.

EEG-MedRAG/
    ├── contexts/                    
        └── knowledge_contexts.json    
    ├── datasets/           
        ├── Epilepsy EEG/                                 
        ├── Parkinson EEG/                           
        ├── Alzheime EEG/                            
        ├── Depression /                             
        ├── Sleep deprivation/
        ├──Mild TBI /
        ├── Psychiatric/
        └── questions.json
    └── openai_api_key.txt 

Step1. Knowledge HyperGraph Construction
nohup python script_insert.py --cls hypertension > result_hypertension_insert.log 2>&1 &
nohup python script_conduct.py --cls hypertension > result_hypertension_insert.log 2>&1 &

Step2. Patient cases HyperGraph Construction
nohup python -u instance_construction/text_entiey.py --output expr/instance_layer/text_entity_list.json 
nohup python -u instance_construction/text_entityvdb.py --output expr/instance_layer/text_entityvdb.json 
nohup python -u instance_construction/hyperedge.py --output expr/instance_layer/hyperedge_list.json
nohup python -u instance_construction/hyperedgevdb.py --output expr/instance_layer/hyperedgevdb.json

Step3.EEG database Construction
nohup python -u instance_construction/EEG_entity.py --output expr/instance_layer/EEG_entity_list.json 
nohup python -u instance_construction/EEG_entityvdb.py --output expr/instance_layer/EEG_entityvdb_list.json 

Step4.Retrieve knowledge,patient cases and EEG of EEG-MedRAG
python retrieval/retrieval_step1.py --output retrieval_step1.json 
python retrieval/retrieval_step2.py --output retrieval_step2.json 
python retrieval/retrieval_step3.py --output retrieval_step3.json
python retrieval/retrieval_step4.py --output retrieval_step4.json  





## Acknowledgement

This repo benefits from [LightRAG](https://github.com/HKUDS/LightRAG), [Text2NKG](https://github.com/LHRLAB/Text2NKG), and [HAHE](https://github.com/LHRLAB/HAHE).  Thanks for their wonderful works.
