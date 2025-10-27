# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, PlainTextResponse
import numpy as np, cv2, io, time
from PIL import Image
from gen.pipeline import GenPipeline

app = FastAPI()
pipe = GenPipeline()

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

def _imread_upload(f: UploadFile) -> np.ndarray:
    data = f.file.read()
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

@app.post("/generate")
def generate(
    selfie: UploadFile = File(...),
    postcard: UploadFile = File(...),
    face: UploadFile = File(...),
    prompt: str = Form(None)
):
    t0 = time.time()
    s_bgr = _imread_upload(selfie)
    p_bgr = _imread_upload(postcard)
    f_bgr = _imread_upload(face)
    out_bgr = pipe.run(s_bgr, p_bgr, f_bgr, prompt=prompt)
    ok, png = cv2.imencode(".png", out_bgr)
    dt = time.time() - t0
    print(f"[timing] io+run={dt:.2f}s, bytes={len(png.tobytes())}", flush=True)
    return Response(content=png.tobytes(), media_type="image/png")
