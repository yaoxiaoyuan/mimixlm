import os
import random
import json
import zipfile
import pyarrow.parquet as pq

instructs = [

"Give me a detailed breakdown of what's happening in this picture.",
"Walk me through everything you can see in the image.",
"Explain every part of the photo as thoroughly as possible.",
"Can you list all the details in this image?",
"Tell me everything that’s going on in the picture.",
"Describe every single thing you notice in the image.",
"Break down what’s shown here—leave nothing out.",
"What’s in this picture? Go into as much detail as you can.",
"Paint a full verbal picture of what this image contains.",
"Share a complete description of everything visible here.",
"Give me a thorough rundown of all elements in the image.",
"What do you observe in this photo? Describe it fully.",
"Lay out every detail you spot in this picture.",
"Take me through every aspect of what’s captured here.",
"Can you catalog everything present in the image?",
"Spell out all the contents of this photo clearly.",
"Provide a comprehensive overview of the image.",
"What’s depicted here? Cover every little thing.",
"Don’t skip anything—describe the entire scene.",
"Walk me through the image inch by inch.",
"Highlight every detail you see in this picture.",
"Give me a meticulous account of the image contents.",
"What’s in the photo? Leave no stone unturned.",
"Narrate everything visible here with precision.",
"Dissect the image and explain all its components.",
"Share a granular description of the picture’s contents.",
"Break down the image—what’s where and how it looks.",
"Describe the photo down to the smallest details.",
"What’s happening in this image? Be exhaustive.",
"Give me a full inventory of what’s in the picture."
]

input_path = "data/LLaVA-NeXT-Data/data/"
input_path_2 = "data/LLaVA-ReCap-CC3M/data/"
output_path = "data/llava_next_convert"
  
if not os.path.exists(output_path): 
    os.makedirs(output_path, exist_ok=True)

remove = 0
fo = open(os.path.join(output_path, "train.jsonl"), "w", encoding="utf-8")

n = 8
seq_id = 0
for f in os.listdir(input_path_2):
    datalist = pq.read_table(os.path.join(input_path_2, f)).to_pylist()
    for data in datalist:
        data["conversations"][0]["value"] = data["conversations"][0]["value"].replace("<image>", "").strip()
        if len(data["conversations"][0]["value"]) == 0:
            data["conversations"][0]["value"] = random.choice(instructs)
        fo.write(json.dumps({"image_path":f"{seq_id}", "conversations":data["conversations"]}, ensure_ascii=False) + "\n")
        seq_id += 1
        if seq_id % 1000 == 0:
            print(f"processed {seq_id} lines")

n = 8
image_fo_list = [zipfile.ZipFile(os.path.join(output_path, f"{i}.zip"), "w") for i in range(n)]
seq_id = 0
for f in os.listdir(input_path):
    datalist = pq.read_table(os.path.join(input_path, f)).to_pylist()
    for data in datalist:
        
        if not data["image"]:
            remove += 1
            continue
         
        data["conversations"][0]["value"] = data["conversations"][0]["value"].replace("<image>", "").strip()

        fo.write(json.dumps({"image_path":f"llava-next-{seq_id}", "conversations":data["conversations"]}, ensure_ascii=False) + "\n") 
        image_fo_list[seq_id%n].writestr(f"llava-next-{seq_id}", data["image"]["bytes"])
        seq_id += 1
        if seq_id % 1000 == 0:
            print(f"done {seq_id} lines, {remove} lines")
fo.close()   
for image_fo in image_fo_list:
    image_fo.close()
