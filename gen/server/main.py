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
    return_debug: str = Form(None),
):
    t0 = time.time()
    s_bgr = _imread_upload(selfie)
    p_bgr = _imread_upload(postcard)
    f_bgr = _imread_upload(face)
    out_bgr = pipe.run(s_bgr, p_bgr, f_bgr, prompt=prompt)

    # Основной результат как PNG
    ok, png = cv2.imencode(".png", out_bgr)
    png_bytes = png.tobytes()

    # Если включён SAVE_DEBUG или явно запрошен return_debug, упаковать zip с результатом и debug_*
    want_debug_bundle = (os.getenv("SAVE_DEBUG", "0") == "1") or (str(return_debug).strip().lower() in {"1", "true", "yes"})
    if want_debug_bundle:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("result.png", png_bytes)
            for path in sorted(glob.glob("/tmp/debug_*")):
                try:
                    with open(path, "rb") as fh:
                        zf.writestr(os.path.basename(path), fh.read())
                except Exception:
                    pass
        buf.seek(0)
        dt = time.time() - t0
        print(f"[timing] io+run={dt:.2f}s, bundle_bytes={len(buf.getvalue())}", flush=True)
        headers = {"Content-Disposition": "attachment; filename=debug_bundle.zip"}
        return Response(content=buf.getvalue(), media_type="application/zip", headers=headers)

    # Обычный путь: отдать только PNG
    dt = time.time() - t0
    print(f"[timing] io+run={dt:.2f}s, bytes={len(png_bytes)}", flush=True)
    return Response(content=png_bytes, media_type="image/png")
