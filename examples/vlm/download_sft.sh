export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --include "data/*.parquet" --repo-type dataset lmms-lab/LLaVA-NeXT-Data --local-dir data/LLaVA-NeXT-Data
