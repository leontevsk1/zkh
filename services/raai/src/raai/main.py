from fastapi import FastAPI
from pathlib import Path
from threading import Lock
import mimetypes
import requests
import os
import json

# ==== Конфиг через env ====
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "audio_in"))
OUT_DIR   = Path(os.getenv("OUT_DIR", "transcripts"))
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8000/transcribe")
# допустимые расширения
EXTS = {".wav"}

# ==== Инициализация ====
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Отсортированный список файлов
file_list = sorted([p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in EXTS])
index = 0
lock = Lock()

app = FastAPI(title="Whisper Controller")

def already_done(p: Path) -> bool:
    # если уже есть файл транскрипта — считаем обработанным
    j = OUT_DIR / (p.stem + ".json")
    t = OUT_DIR / (p.stem + ".txt")
    return j.exists() and t.exists()

def mime_for(p: Path) -> str:
    return mimetypes.guess_type(str(p))[0] or "application/octet-stream"

def post_to_whisper(path: Path, timeout=120):
    with path.open("rb") as f:
        files = {"file": (path.name, f, mime_for(path))}
        r = requests.post(WHISPER_URL, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()

def save_result(src: Path, payload: dict):
    (OUT_DIR / (src.stem + ".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text = (payload.get("text") or "").strip()
    (OUT_DIR / (src.stem + ".txt")).write_text(text, encoding="utf-8")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "whisper_url": WHISPER_URL, "total": len(file_list)}

@app.get("/next")
def process_next():
    global index

    with lock:
        # пропускаем уже обработанные
        while index < len(file_list) and already_done(file_list[index]):
            index += 1

        if index >= len(file_list):
            return {"status": "END", "processed": index, "total": len(file_list)}

        path = file_list[index]
        idx = index
        index += 1

    # синхронно отправляем файл в Whisper и ждём ответ
    try:
        payload = post_to_whisper(path)
        save_result(path, payload)
        return {
            "status": "OK",
            "file": str(path.name),
            "index": idx,
            "remaining": len(file_list) - index,
            "text_preview": (payload.get("text") or "")[:120]
        }
    except requests.HTTPError as e:
        return {"status": "WHISPER_HTTP_ERROR", "file": path.name, "detail": str(e)}
    except requests.RequestException as e:
        return {"status": "WHISPER_NETWORK_ERROR", "file": path.name, "detail": str(e)}
    except Exception as e:
        return {"status": "ERROR", "file": path.name, "detail": str(e)}
