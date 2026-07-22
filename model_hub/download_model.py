from huggingface_hub import snapshot_download

model_id = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
local_dir = "/home/lzg/zyt/models/Llama-3-8B-Instruct-Gradient-1048k"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)