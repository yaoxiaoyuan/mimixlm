export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --include "sample/10BT/*.parquet"  --repo-type dataset HuggingFaceFW/fineweb-edu --local-dir data/fineweb/
huggingface-cli download --resume-download  --include "20231101.en/*" --repo-type dataset wikimedia/wikipedia --local-dir data/wiki_en/

