# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, PlainTextResponse
import numpy as np, cv2, io, time, os, glob, zipfile
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
    prompt: str = Form(None),
):
    t0 = time.time()
    s_bgr = _imread_upload(selfie)
    p_bgr = _imread_upload(postcard)
    f_bgr = _imread_upload(face)
    out_bgr = pipe.run(s_bgr, p_bgr, f_bgr, prompt=prompt)

    # Основной результат как PNG
    ok, png = cv2.imencode(".png", out_bgr)
    png_bytes = png.tobytes()

    # Обычный путь: отдать только PNG
    dt = time.time() - t0
    print(f"[timing] io+run={dt:.2f}s, bytes={len(png_bytes)}", flush=True)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/debug-bundle")
def debug_bundle(clear: bool = False):
    files = sorted(glob.glob("/tmp/debug_*"))
    if not files:
        return PlainTextResponse("no debug files found; enable SAVE_DEBUG=1 and run /generate", status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            try:
                with open(path, "rb") as fh:
                    zf.writestr(os.path.basename(path), fh.read())
            except Exception:
                pass
    if clear:
        for path in files:
            try:
                os.remove(path)
            except Exception:
                pass
    buf.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=debug_bundle_{int(time.time())}.zip"}
    return Response(content=buf.getvalue(), media_type="application/zip", headers=headers)
