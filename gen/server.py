import io, os, time, traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from PIL import Image
import numpy as np
import cv2

from pipeline import GenPipeline

app = FastAPI()

def _bgr_from_upload(f: UploadFile) -> np.ndarray:
    data = f.file.read()
    if not data:
        raise HTTPException(400, f"Empty file: {f.filename}")
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(400, f"Bad image {f.filename}: {e}")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/")
def root():
    return PlainTextResponse("ok", 200)

# --- инициализация пайплайна
pipe = GenPipeline()

# --- опциональный warmup (на CPU по умолчанию пропускаем, чтобы не ловить OOM)
if os.getenv("SKIP_WARMUP","1") != "1" and os.getenv("USE_CPU","0") != "1":
    try:
        h, w = 128, 128
        dummy = np.zeros((h, w, 3), np.uint8)
        pipe.run(dummy.copy(), dummy.copy(), dummy.copy(), prompt="")
    except Exception:
        pass

@app.post("/generate")
def generate(
    selfie: UploadFile = File(...),
    postcard: UploadFile = File(...),
    face: UploadFile = File(...)
):
    t0 = time.time()
    try:
        selfie_bgr   = _bgr_from_upload(selfie)
        postcard_bgr = _bgr_from_upload(postcard)
        face_bgr     = _bgr_from_upload(face)

        # мягкий даунскейл входов
        max_side_env = int(os.getenv("MAX_SIDE", "1024"))
        def _resize_limit(img, smax):
            h,w = img.shape[:2]
            if max(h,w) <= smax: return img
            r = smax / max(h,w)
            return cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)
        selfie_bgr   = _resize_limit(selfie_bgr,   max_side_env)
        postcard_bgr = _resize_limit(postcard_bgr, max_side_env)
        face_bgr     = _resize_limit(face_bgr,     max_side_env)

        t1 = time.time()
        out_bgr = pipe.run(selfie_bgr, postcard_bgr, face_bgr)
        t2 = time.time()

        ext = (os.getenv("OUTPUT_FORMAT","png")).lower()
        if ext not in {"jpg","jpeg","png","webp"}: ext = "png"
        im = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        save_kwargs = {}
        if ext in ("jpg","jpeg"):
            save_kwargs["quality"] = int(os.getenv("JPEG_QUALITY","92"))
            save_kwargs["optimize"] = True
        im.save(buf, format=ext.upper(), **save_kwargs)
        body = buf.getvalue()

        print(f"[timing] io={(t1-t0):.2f}s, run={(t2-t1):.2f}s, total={(time.time()-t0):.2f}s, bytes={len(body)}", flush=True)
        return Response(body, media_type=f"image/{'jpeg' if ext=='jpg' else ext}")

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("[error] ", tb, flush=True)
        # Возвращаем JSON 500 вместо разрыва сокета
        return JSONResponse({"error": str(e)}, status_code=500)
