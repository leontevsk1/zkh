import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import requests
import time
import uvicorn

# --- Конфиг через env ---
MODEL_NAME   = os.getenv("MODEL_NAME", "small")
DEVICE       = os.getenv("DEVICE", "cpu") # "cpu" | "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
NUM_THREADS  = int(os.getenv("NUM_THREADS", "8"))
NEXT_SERVICE_URL = os.getenv("NEXT_SERVICE_URL")
CALLBACK_TIMEOUT = float(os.getenv("CALLBACK_TIMEOUT", "5"))
CALLBACK_RETRIES = int(os.getenv("CALLBACK_RETRIES", "3"))

# --- Инициализация модели ---
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    num_threads=NUM_THREADS
)

app = FastAPI(title="faster-whisper API (ru)")

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "threads": NUM_THREADS
    }

def _post_with_retry(url: str, payload: dict, timeout: float, retries: int):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return {"ok": True, "status_code": r.status_code}
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5 * (2 ** (attempt - 1)))  # экспоненциальная пауза
    return {"ok": False, "error": last_err}

@app.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_callback_url: Optional[str] = Header(default=None, convert_underscores=False),
    x_request_id: Optional[str] = Header(default=None, convert_underscores=False),
):
    callback_url = x_callback_url or NEXT_SERVICE_URL

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp_path = tmp.name
    finally:
        tmp.close()

    try:
        segments, info = model.transcribe(
            tmp_path,
            language="ru",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )

        seg_list: List[dict] = []
        full_text_parts = []
        for s in segments:
            seg_list.append({"start": s.start, "end": s.end, "text": s.text})
            full_text_parts.append(s.text)
        text = " ".join(full_text_parts).strip()

        payload = {
            "request_id": x_request_id,
            "filename": file.filename,
            "language": info.language,
            "duration": info.duration,
            "text": text,
            "segments": seg_list,
            "model": MODEL_NAME,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        }

        if callback_url and text:
            def _bg_send(url, data):
                res = _post_with_retry(url, data, CALLBACK_TIMEOUT, CALLBACK_RETRIES)
                if not res.get("ok"):
                    data["forward_error"] = res.get("error")

            background_tasks.add_task(_bg_send, callback_url, payload)

        return JSONResponse(payload)

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# --- Автозапуск сервера ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
