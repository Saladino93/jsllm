from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='~/models/Qwen2.5-7B-Instruct')
snapshot_download('jane-street/dormant-model-warmup', local_dir='~/models/dormant-model-warmup')
