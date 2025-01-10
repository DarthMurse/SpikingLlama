export HF_ENDPOINT=https://hf-mirror.com
#python3 scripts/prepare_slimpajama.py --split=validation --percentage 1.0
python3 -m scripts.prepare_openwebtext
