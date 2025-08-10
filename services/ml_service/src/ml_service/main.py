import joblib
from fastapi import FastAPI
from pydantic import BaseModel

svc = joblib.load("svc_clf.joblib")
pri = joblib.load("priority_clf.joblib")

app = FastAPI()

class Inp(BaseModel):
    text: str

@app.post("/infer")
def infer(inp: Inp):
    svc_pred = svc.predict([inp.text])[0]
    pri_pred = pri.predict([f"__svc__{svc_pred} {inp.text}"])[0]
    prob = problem_extract(inp.text, topk=1)
    return {
        "service": svc_pred,
        "priority": pri_pred,
        "problem": prob
    }