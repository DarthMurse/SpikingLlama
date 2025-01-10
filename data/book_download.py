def book_download():
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="bookcorpus", repo_type="dataset", resume_download=True, local_dir="bookcorpus", local_dir_use_symlinks=True)
