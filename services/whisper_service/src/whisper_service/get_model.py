import os
from huggingface_hub import snapshot_download

REPO = os.environ.get("FW_REPO", "Systran/faster-whisper-small")
OUT  = os.environ.get("FW_DIR",  "./models/faster-whisper-small")



# ускорение скачивания (нужен пакет hf_transfer)
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer  # noqa: F401
        print("hf_transfer enabled")
    except Exception:
        print("⚠️ hf_transfer не установлен, продолжу обычным способом")

path = snapshot_download(
    repo_id=REPO,
    local_dir=OUT,
    local_dir_use_symlinks=False,  # игнорируется в новых версиях, но не мешает
    resume_download=True,          # докачка с места обрыва
    max_workers=8,                 # параллельные потоки
    allow_patterns=["*.json", "model.bin", "vocabulary*","tokenizer*"] 
)

print("Downloaded to:", path)
