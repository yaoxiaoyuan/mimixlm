# Configure HuggingFace endpoint mirror for China mainland users 
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download --include "data/*.parquet" --repo-type dataset roneneldan/TinyStories --local-dir data/TinyStories
