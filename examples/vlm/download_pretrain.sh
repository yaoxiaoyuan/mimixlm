export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --include "data/*.parquet" --repo-type dataset lmms-lab/LLaVA-ReCap-CC3M --local-dir data/LLaVA-ReCap-CC3M

