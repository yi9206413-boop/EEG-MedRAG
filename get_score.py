import json
import os
from evaluation.eval import cal_em, cal_f1
from evaluation.eval_r import cal_rsim
from evaluation.eval_g import cal_gen
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor

BASELINE = " "    
MODEL = " "         

METHOD = f"{BASELINE}_{MODEL}"
def evaluate_one(d):
    try:
        generation = d['generation']
        try:
            answer = generation.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            answer = generation
            
        # if "Yes" in d['golden_answers']:
        #     d['golden_answers'] = "Yes"
        # elif "No" in d['golden_answers']:
        #     d['golden_answers'] = "No"
            
        if "Yes" in answer:
            answer = "Yes"
        elif "No" in answer:
            answer = "No"
            
        em_score = cal_em([d['golden_answers']], [answer])
        f1_score = cal_f1([d['golden_answers']], [answer])
        # gen_score = cal_gen(d['question'], d['golden_answers'], generation, f1_score)

        d['em'] = em_score
        d['f1'] = f1_score
        # d['gen'] = gen_score["score"]
        # d['gen_exp'] = gen_score["explanation"]

        return d
    except Exception as e:
        print(f"[ERROR] Failed processing sample: {d.get('question', 'N/A')}")
        traceback.print_exc()
        raise

def evaluate_method():
    success_flag = False  # 控制是否成功保存

    try:
        print(f"[DEBUG] test_generation.json")
        data_dir = f"test_generation_{METHOD}.json"
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"File not found: {data_dir}")

        with open(data_dir) as f:
            data = json.load(f)

        # 并行处理样本
        max_workers = 64
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            data = list(tqdm(executor.map(evaluate_one, data), total=len(data), desc="EEG-RAG"))

        # 汇总指标
        overall_em = sum([d['em'] for d in data])*100 / len(data)
        overall_f1 = sum([d['f1'] for d in data])*100 / len(data)
        # overall_gen = sum([d['gen'] for d in data]) / len(data)

        print(f"Overall EM: {overall_em:.2f}")
        print(f"Overall F1: {overall_f1:.2f}")
        # print(f"Overall Gen: {overall_gen:.4f}")
        
        domain_score = {}
        task_score = {}
        role_score = {}
        
        for d in data:
            if d["domain"] not in domain_score:
                domain_score[d["domain"]]={"EM":[],"F1":[]}
            if d["task"] not in task_score:
                task_score[d["task"]]={"EM":[],"F1":[]}
            if d["role"] not in role_score:
                role_score[d["role"]]={"EM":[],"F1":[]}
            domain_score[d["domain"]]["EM"].append(d['em']*100)
            domain_score[d["domain"]]["F1"].append(d['f1']*100)
            task_score[d["task"]]["EM"].append(d['em']*100)
            task_score[d["task"]]["F1"].append(d['f1']*100)     
            role_score[d["role"]]["EM"].append(d['em']*100)
            role_score[d["role"]]["F1"].append(d['f1']*100)
        
        # print("Domains:")
        # for domain in domain_score:
        #     print(f"{domain}_EM: {sum(domain_score[domain]['EM']) / len(domain_score[domain]['EM']):.2f}")
        #     print(f"{domain}_F1: {sum(domain_score[domain]['F1']) / len(domain_score[domain]['F1']):.2f}")
        # print()
        # print("Tasks:")
        # for task in task_score:
        #     print(f"{task}_EM: {sum(task_score[task]['EM']) / len(task_score[task]['EM']):.2f}")
        #     print(f"{task}_F1: {sum(task_score[task]['F1']) / len(task_score[task]['F1']):.2f}")
        # print()
        # print("Roles:") 
        # for role in role_score:
        #     print(f"{role}_EM: {sum(role_score[role]['EM']) / len(role_score[role]['EM']):.2f}")
        #     print(f"{role}_F1: {sum(role_score[role]['F1']) / len(role_score[role]['F1']):.2f}")
        # print()
        
        from tabulate import tabulate
        save_base = f"scores/{METHOD}"
        os.makedirs(save_base, exist_ok=True)

        # 打印 Domains 表格
        print("Domains:")
        domain_table = []
        for domain in domain_score:
            em = sum(domain_score[domain]['EM']) / len(domain_score[domain]['EM'])
            f1 = sum(domain_score[domain]['F1']) / len(domain_score[domain]['F1'])
            domain_table.append([domain, f"{em:.2f}", f"{f1:.2f}"])
        domain_table_str = tabulate(domain_table, headers=["Domain", "EM", "F1"], tablefmt="grid")
        print(tabulate(domain_table, headers=["Domain", "EM", "F1"], tablefmt="grid"))
        print()
        
        with open(os.path.join(save_base, "domain_table.txt"), 'w') as f:
               f.write(domain_table_str)


        # 打印 Tasks 表格
        print("Tasks:")
        task_table = []
        for task in task_score:
            em = sum(task_score[task]['EM']) / len(task_score[task]['EM'])
            f1 = sum(task_score[task]['F1']) / len(task_score[task]['F1'])
            task_table.append([task, f"{em:.2f}", f"{f1:.2f}"])
        print(tabulate(task_table, headers=["Task", "EM", "F1"], tablefmt="grid"))
        print()
        task_table_str = tabulate(task_table, headers=["Task", "EM", "F1"], tablefmt="grid")
        with open(os.path.join(save_base, "task_table.txt"), 'w') as f:
              f.write(task_table_str)

        # 打印 Roles 表格
        print("Roles:")
        role_table = []
        for role in role_score:
            em = sum(role_score[role]['EM']) / len(role_score[role]['EM'])
            f1 = sum(role_score[role]['F1']) / len(role_score[role]['F1'])
            role_table.append([role, f"{em:.2f}", f"{f1:.2f}"])
        print(tabulate(role_table, headers=["Role", "EM", "F1"], tablefmt="grid"))
        print()
        role_table_str = tabulate(role_table, headers=["Role", "EM", "F1"], tablefmt="grid")
        with open(os.path.join(save_base, "role_table.txt"), 'w') as f:
               f.write(role_table_str)


        save_base = f"scores/{METHOD}"
        os.makedirs(save_base, exist_ok=True)

        result_path = os.path.join(save_base, "test_result.json")
        with open(result_path, 'w') as f:
            json.dump(data, f, indent=4)

        score_path = os.path.join(save_base, "test_score.json")
        with open(score_path, 'w') as f:
            json.dump({
                "overall_em": overall_em,
                "overall_f1": overall_f1,
                # "overall_gen": overall_gen,
            }, f, indent=4)

        # 成功保存标志
        success_flag = True
        print(f"[SAVED] {result_path}")
        print(f"[SAVED] {score_path}")
        print(f"[SUCCESS] finished and saved.")

    except Exception as e:
        print(f"\n[ERROR] failed due to: {str(e)}")
        traceback.print_exc()
        raise

    if not success_flag:
        raise RuntimeError(f"did not complete saving.")
    
    return True

if __name__ == "__main__":
    evaluate_method()
