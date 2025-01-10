def slim_download():
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="cerebras/SlimPajama-627B", repo_type="dataset", resume_download=True, local_dir="SlimPajama-627B", local_dir_use_symlinks=True)
