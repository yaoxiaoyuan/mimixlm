export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download openai/clip-vit-large-patch14 --local-dir model/clip-vit-large-patch14
python examples/convert_hf/convert_hf_weights.py --hf_path model/clip-vit-large-patch14 --convert_path model/vlm/init/
cp model/gpt_large_2048/model_weights model/vlm/init/model_weights_1
cp model/gpt_large_2048/tokenizer.model model/vlm/init/
cp examples/vlm/model_config.json model/vlm/init/
python mimixlm.py --mode init --init_config_path model/vlm/init/model_config.json --init_model_path model/vlm/init --init_weight_path model/vlm/init/model_weights_0  model/vlm/init/model_weights_1
rm model/vlm/init/model_weights_0  model/vlm/init/model_weights_1
