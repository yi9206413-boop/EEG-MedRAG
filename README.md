# EEG-MedRAG

EEG-MedRAG:EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation [[paper](https://arxiv.org/abs/2503.21322)]

##  Overview 

<img width="765" height="389" alt="b0a11d20-41ba-477b-bba0-cd5dcce5268f" src="https://github.com/user-attachments/assets/47110688-e487-4f3f-9557-7cf40d8f7e9c" />


## Environment Setup
```bash
conda create python=3.11
conda activate ptthon3.11
pip install -r requirements.txt(https://github.com/user-attachments/files/21887616/F3.pdf)

```

Then, we need to set openai api key in openai_api_key.txt file. (We use www.apiyi.com for LLM server.)

Last, we need download the contexts and datasets from https://drive.google.com/drive/folders/1tPjD1Om2qp-fSPlxbwBP5RlDHw4LaRRm?usp=sharing, https://physionet.org/content/chbmit/1.0.0/ and https://openneuro.org/ and put them in the contexts and datasets folders.

EEG-MedRAG/
    ├── contexts/                    
        └── knowledge_contexts.json/    
    ├── datasets/           
        ├── Epilepsy EEG/                                 
        ├── Parkinson EEG/                           
        ├── Alzheime EEG/                            
        ├── Depression /                             
        ├── Sleep deprivation/
        ├──Mild TBI /
        ├── Psychiatric/
        └── questions.json/
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



If you find this work is helpful for your research, please cite:
@misc{wang2025eegmedragenhancingeegbasedclinical,
      title={EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation}, 
      author={Yi Wang and Haoran Luo and Lu Meng},
      year={2025},
      eprint={2508.13735},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.13735}, 
}





## Acknowledgement

This repo benefits from [HypergraphRAG](https://github.com/LHRLAB/HyperGraphRAG), [DHGE](https://github.com/LHRLAB/DHGE).  Thanks for their wonderful works.
