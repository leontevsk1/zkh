from pathlib import Path
import os
import time
import json
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
from huggingface_hub import snapshot_download

from pydantic import BaseModel
from fastapi import FastAPI, Request
import uvicorn
import joblib

# === Paths ===
MODULES_DIR = Path(__file__).resolve().parent.parent.parent / "modules"
ADAPTER_DIR = MODULES_DIR / "zkh_problem_lora"   # распакованный адаптер LoRA (папка)

# Можно стартовать и без адаптера (например, если используешь merged-модель)
USE_LORA = ADAPTER_DIR.exists()

# === База из адаптера или ENV ===
base_model_name = os.getenv("PROBLEM_BASE_MODEL", None)
if USE_LORA:
    peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
    base_model_name = base_model_name or peft_cfg.base_model_name_or_path

if not base_model_name:
    # Фоллбек, если ни адаптера ни ENV
    base_model_name = "cointegrated/rut5-small"

# === Локальный кеш базы внутри проекта ===
BASE_CACHE_DIR = MODULES_DIR / "base_models" / base_model_name.replace("/", "__")

# === Таймауты/ускорители загрузки с HF (если вдруг разрешено) ===
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

def ensure_base_model_local(repo_id: str, local_dir: Path) -> Path:
    """
    Если local_dir/config.json существует — считаем, что база уже разложена офлайн.
    Иначе пробуем скачать (в онлайне). В офлайне упадёт с ясной ошибкой.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    if (local_dir / "config.json").exists():
        return local_dir

    last_err = None
    for attempt in range(3):
        try:
            p = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=[
                    "*.json", "*.safetensors", "*.bin",
                    "tokenizer.*", "spiece.model", "vocab.*", "merges.txt"
                ],
            )
            return Path(p)
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(
        f"Не удалось получить базовую модель {repo_id} -> {local_dir}. "
        f"Разложи архив офлайн. Последняя ошибка: {last_err}"
    )

# === Гарантируем локальную копию базы (офлайн папка с config.json) ===
local_base = ensure_base_model_local(base_model_name, BASE_CACHE_DIR)

# === Токенайзер ===
try:
    tokenizer = AutoTokenizer.from_pretrained(local_base, use_fast=True, legacy=False)
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(local_base, use_fast=True)
except Exception:
    # fallback: попробуем взять токенайзер из адаптера (если там лежит кастомный)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)

# === Модель ===
if USE_LORA:
    base = AutoModelForSeq2SeqLM.from_pretrained(local_base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
else:
    # Поддержка сценария с merged-моделью (когда база уже содержит LoRA-веса)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_base, torch_dtype=torch.float32)

model.eval()

@torch.inference_mode()
def t5_summarize(text: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# === Загрузка классификаторов и метаданных (фоллбеки) ===
def _load_or_dummy_clf(path_joblib: Path, n_classes: int, class_labels=None):
    if path_joblib.exists():
        return joblib.load(path_joblib)

    class _DummyClf:
        def __init__(self, labels):
            self.classes_ = np.array(labels)

        def predict_proba(self, X):
            n = len(X)
            probs = np.zeros((n, len(self.classes_)), dtype=float)
            probs[:, 0] = 1.0
            return probs

    if class_labels is None:
        class_labels = list(range(n_classes))
    return _DummyClf(class_labels)

def _load_json_or_default(path_json: Path, default: dict):
    try:
        if path_json.exists():
            return json.loads(path_json.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

SERVICE_CLF_PATH   = MODULES_DIR / "service_clf.joblib"
URGENCY_CLF_PATH   = MODULES_DIR / "urgency_clf.joblib"
SERVICE_META_PATH  = MODULES_DIR / "service_meta.json"
URGENCY_META_PATH  = MODULES_DIR / "urgency_meta.json"

service_clf = _load_or_dummy_clf(SERVICE_CLF_PATH, n_classes=1, class_labels=["other"])
svc_meta    = _load_json_or_default(SERVICE_META_PATH, {"version": "unknown", "source": "dummy"})

urgency_clf = _load_or_dummy_clf(URGENCY_CLF_PATH, n_classes=4, class_labels=[0,1,2,3])
urgency_meta = _load_json_or_default(URGENCY_META_PATH, {"version": "unknown", "source": "dummy"})

# === Pydantic модели ===
class MLRequest(BaseModel):
    task_id: str
    text: str

class MLResponse(BaseModel):
    task_id: str
    service: dict
    urgency: dict
    problem: dict

class InferIn(BaseModel):
    text: str

# === Инференс ===
@torch.inference_mode()
def run_inference(text: str):
    # Service classification
    svc_proba = service_clf.predict_proba([text])[0]
    svc_idx = int(np.argmax(svc_proba))
    svc_label = service_clf.classes_[svc_idx]
    service_result = {
        "label": str(svc_label),
        "proba": float(svc_proba[svc_idx]),
        "model_meta": svc_meta,
    }

    # Urgency prediction
    urg_proba = urgency_clf.predict_proba([text])[0]
    urg_idx = int(np.argmax(urg_proba))
    urgency_result = {
        "class": int(urg_idx) + 1,  # 1..4
        "proba": float(urg_proba[urg_idx]),
        "model_meta": urgency_meta,
    }

    # Problem summarization
    summary = t5_summarize(text, max_new_tokens=50)
    problem_result = {
        "summary": summary,
        "model_meta": {
            "type": "lora" if USE_LORA else "merged",
            "base_model": str(local_base),
            "adapter_dir": str(ADAPTER_DIR) if USE_LORA else None,
        },
    }

    return service_result, urgency_result, problem_result

# === FastAPI ===
app = FastAPI()

# Совместимость с твоим старым контрактом
@app.post("/run", response_model=MLResponse)
def run_ml(req: MLRequest):
    service_result, urgency_result, problem_result = run_inference(req.text)
    return MLResponse(
        task_id=req.task_id,
        service=service_result,
        urgency=urgency_result,
        problem=problem_result,
    )

# Контракт под RAAI: POST /infer с {"text": "..."}
@app.post("/infer")
def infer(in_: InferIn, request: Request):
    service_result, urgency_result, problem_result = run_inference(in_.text)
    # вернём id из заголовка, если есть (RAAI шлёт X-Request-Id)
    rid = request.headers.get("X-Request-Id")
    resp = {
        "request_id": rid,
        "service": service_result,
        "urgency": urgency_result,
        "problem": problem_result,
    }
    return resp

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# === Entrypoint ===
if __name__ == "__main__":
    # В RAAI ожидается порт 8010
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False)
