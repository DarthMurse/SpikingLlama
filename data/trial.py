from datasets import load_dataset
import datasets
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com" # 需要设置一下环境变量
#os.environ["HF_DATASETS_CACHE"] = "/data/hf_dataset/"
config = datasets.DownloadConfig(resume_download=True, max_retries=100)
dataset = datasets.load_dataset("cerebras/SlimPajama-627B", download_config=config)

