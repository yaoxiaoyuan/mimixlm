python examples/vlm/convert_LLaVA-ReCap-CC3M.py
python mimixlm.py --task process-llava-recap-3m --mode preprocess --stage pretrain --model_path model/gpt_large_2048/ --raw_data_path data/llava_recap_convert/train.jsonl --processed_data_path data/llava_recap_processed --n_preprocess_workers 8 --max_len 2048 --n_split_shards 8 --text_fields text --image_path_fields 'image_path'
