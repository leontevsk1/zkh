from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from uuid import uuid4
from pathlib import Path
from threading import Lock
import requests, time
import uvicorn

ASR_URL = "http://whisper:8000/transcribe"
ML_URL  = "http://ml:8010/infer"
ASR_LIMIT = 1
ML_LIMIT  = 1

app = FastAPI()
lock = Lock()

jobs = {}  # id -> dict
q_asr, q_ml = [], []
asr_inflight = 0
ml_inflight  = 0
AUDIO_DIR = Path("data/audio"); AUDIO_DIR.mkdir(parents=True, exist_ok=True)

class MLCallback(BaseModel):
    request_id: str
    service: str
    priority: str
    problem: str

def schedule():
    global asr_inflight, ml_inflight
    with lock:
        while asr_inflight < ASR_LIMIT and q_asr:
            jid = q_asr.pop(0)
            _send_to_asr(jid)
        while ml_inflight < ML_LIMIT and q_ml:
            jid = q_ml.pop(0)
            _send_to_ml(jid)

def _post_json(url, json, headers=None, retries=3, timeout=10):
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, json=json, headers=headers, timeout=timeout)
            r.raise_for_status()
            return {"ok": True, "status": r.status_code}
        except Exception as e:
            last = str(e); time.sleep(0.5 * (2**i))
    return {"ok": False, "error": last}

def _send_to_asr(jid):
    global asr_inflight
    job = jobs[jid]
    if job["status"] not in ("NEW", "QUEUED_ASR"):
        return
    job["status"] = "SENT_TO_ASR"
    asr_inflight += 1
    files = {"file": open(job["wav_path"], "rb")}
    headers = {"X-Request-Id": jid, "X-Callback-Url": "http://controller:8000/cb/asr"}
    try:
        r = requests.post(ASR_URL, files=files, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        job["last_error"] = str(e)
        job["status"] = "QUEUED_ASR"
        asr_inflight -= 1
        q_asr.append(jid)
    finally:
        try: files["file"].close()
        except: pass

def _send_to_ml(jid):
    global ml_inflight
    job = jobs[jid]
    if job["status"] not in ("ASR_DONE", "QUEUED_ML"):
        return
    job["status"] = "SENT_TO_ML"
    ml_inflight += 1
    payload = {"text": job["asr_text"]}
    headers = {"X-Request-Id": jid, "X-Callback-Url": "http://controller:8000/cb/ml"}
    res = _post_json(ML_URL, payload, headers=headers, timeout=15, retries=3)
    if not res["ok"]:
        job["last_error"] = res["error"]
        job["status"] = "QUEUED_ML"
        ml_inflight -= 1
        q_ml.append(jid)

@app.post("/ingest_audio")
async def ingest_audio(file: UploadFile = File(...)):
    jid = str(uuid4())
    wav_path = AUDIO_DIR / f"{jid}.wav"
    wav_path.write_bytes(await file.read())

    with lock:
        jobs[jid] = {"id": jid, "status":"NEW", "wav_path": str(wav_path),
                     "asr_text": None, "ml_result": None, "last_error": None}
        # планирование на ASR
        if asr_inflight < ASR_LIMIT:
            _send_to_asr(jid)
        else:
            jobs[jid]["status"] = "QUEUED_ASR"
            q_asr.append(jid)
    schedule()
    return {"request_id": jid}

@app.post("/cb/asr")
async def cb_asr(payload: dict):
    jid = payload.get("request_id")
    text = payload.get("text", "")
    with lock:
        job = jobs.get(jid)
        if not job:
            return {"status":"unknown"}
        if job["status"] == "ASR_DONE":
            return {"status":"ok"}  # идемпотентность
        job["asr_text"] = text
        job["status"] = "ASR_DONE"
        # освободить слот ASR
        global asr_inflight; asr_inflight = max(0, asr_inflight - 1)
        # поставить в ML
        if ml_inflight < ML_LIMIT:
            _send_to_ml(jid)
        else:
            job["status"] = "QUEUED_ML"
            q_ml.append(jid)
    schedule()
    return {"status":"ok"}

@app.post("/cb/ml")
async def cb_ml(cb: MLCallback):
    jid = cb.request_id
    with lock:
        job = jobs.get(jid)
        if not job:
            return {"status":"unknown"}
        if job["status"] == "ML_DONE":
            return {"status":"ok"}  # идемпотентность
        job["ml_result"] = cb.dict()
        job["status"] = "ML_DONE"
        global ml_inflight; ml_inflight = max(0, ml_inflight - 1)
    # тут можно вызвать completion-service
    schedule()
    return {"status":"ok"}

@app.get("/status/{jid}")
def get_status(jid: str):
    return jobs.get(jid, {"status":"not-found"})

@app.get("/healthz")
def healthz():
    with lock:
        return {"asr_inflight": asr_inflight, "ml_inflight": ml_inflight,
                "q_asr": len(q_asr), "q_ml": len(q_ml), "jobs": len(jobs)}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)