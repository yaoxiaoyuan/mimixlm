import os
import json
import random
import pyarrow.parquet as pq

paths = [
        "data/MT-SFT-ShareGPT/en",
        "data/wildchat/data",
        "data/Infinity-Instruct/7M",        
        "data/Infinity-Instruct/Gen",
        "data/Magpie-Llama-3.1-Pro-MT-300K-Filtered/data",
        "data/smoltalk/data/all",
        "data/OpenHermes-2.5",
        "data/OpenOrca",
        "data/ultrachat_200k/data",
        "data/tulu-v2-sft-mixture/data" 
    ]

lines = set()
for path in paths:
    for f in os.listdir(path):
        if f.endswith(".jsonl"):
            data = [json.loads(line) for line in open(os.path.join(path, f), "rb")]
        elif f.endswith(".parquet"):
            data = pq.read_table(os.path.join(path, f)).to_pylist()        
        elif f.endswith(".json"):
            data = json.load(open(os.path.join(path, f), "rb")) 
        else:
            continue

        for d in data:
            if "language" in d and d["language"] not in ["English", "EN"]:
                continue
            if "langdetect" in d and d["langdetect"] not in ["en"]:
                continue

            if "conversation" in d:
                d = d["conversation"] 
            elif "conversations" in d:
                d = d["conversations"]
            elif "messages" in d:
                d = d["messages"]
            elif "system_prompt" in d and "question" in d and "response" in d:
                d = [{"role":"system", "content":d["system_prompt"]},
                     {"role":"user", "content":d["question"]},
                     {"role":"assistant", "content":d["response"]}]
            else:
                print(path,d, "fail")
                continue

            norm_d = []
            for dd in d:
                role = dd.get("role", "")
                if not role:
                    role = dd.get("from", "")
                if role in ["human", "user"]:
                    role = "user"
                elif role in ["assistant", "gpt"]:
                    role = "assistant"
                elif role in ["system"]:
                    role = "system"
                else:
                    print("err!", dd)

                content = dd.get("content", "")
                if not content:
                    content = dd.get("value", "")
                if not content:
                    print("warn!", dd)

                norm_d.append({"role":role, "content": content})

            out = json.dumps({"conversations": norm_d}, ensure_ascii=False) + "\n"
 
            if out in lines:
                print(path, "duplicate")
            else:
                lines.add(out)
                print(path, len(lines))

lines = list(lines)
random.shuffle(lines)

if not os.path.exists("data/sft_mix/"):
    os.makedirs("data/sft_mix/", exist_ok=True)

fo = open("data/sft_mix/merge_sft_data.jsonl", "w", encoding="utf-8")      
for line in lines:
    fo.write(line)
fo.close()




