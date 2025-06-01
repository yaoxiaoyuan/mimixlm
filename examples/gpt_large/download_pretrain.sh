export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --include "sample/100BT/*.parquet"  --repo-type dataset HuggingFaceFW/fineweb --local-dir data/fineweb
