def star_download():
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    from huggingface_hub import snapshot_download
    import huggingface_hub
    huggingface_hub.login("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")
    snapshot_download(repo_id="bigcode/starcoderdata", repo_type="dataset", resume_download=True, local_dir="starcoderdata", local_dir_use_symlinks=True)
