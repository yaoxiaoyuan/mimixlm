import os
import json
import zipfile
import pyarrow.parquet as pq

input_path = "data/LLaVA-ReCap-CC3M/data/"
output_path = "data/llava_recap_convert"
 
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

fo = open(os.path.join(output_path, "train.jsonl"), "w", encoding="utf-8")
n = 8
image_fo_list = [zipfile.ZipFile(os.path.join(output_path, f"{i}.zip"), "w") for i in range(n)]
seq_id = 0
for f in os.listdir(input_path):
    datalist = pq.read_table(os.path.join(input_path, f)).to_pylist()
    for data in datalist:
        fo.write(json.dumps({"image_path":str(seq_id), "text":data["conversations"][-1]["value"]}, ensure_ascii=False) + "\n") 
        image_fo_list[seq_id%n].writestr(str(seq_id), data["image"]["bytes"])
        seq_id += 1
        if seq_id % 1000 == 0:
            print(f"processed {seq_id} lines")
fo.close()   
for image_fo in image_fo_list:
    image_fo.close()
