python examples/vlm/convert_LLaVA-NeXT-Data.py
python mimixlm.py --task process-llava-merge --mode preprocess --stage sft --model_path model/gpt_large_2048/ --raw_data_path data/llava_next_convert/train.jsonl --processed_data_path data/llava_merge_processed --n_preprocess_workers 8 --max_len 2048 --n_split_shards 8 --conversation_fields conversations --image_path_fields 'image_path'
