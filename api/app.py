from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, Literal
import io, os
import cv2
import numpy as np
from processing import (
    compose_postcard_on_selfie,
    reinhard_color_transfer, histogram_match_bgr,  # экспортируем на будущее
)

app = FastAPI(title="Selfie+Postcard API (MVP)", version="0.1.0")

def env(name, default):
    return os.getenv(name, default)

DEFAULT_COLOR = env("COLOR", "reinhard")
DEFAULT_COLOR_STRENGTH = float(env("COLOR_STRENGTH", "0.8"))
DEFAULT_SHADOW_STRENGTH = float(env("SHADOW_STRENGTH", "0.15"))
DEFAULT_MAX_SIZE = int(env("MAX_SIZE", "1600"))
DEFAULT_JPEG_QUALITY = int(env("JPEG_QUALITY", "92"))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "defaults": dict(
            color=DEFAULT_COLOR,
            color_strength=DEFAULT_COLOR_STRENGTH,
            shadow_strength=DEFAULT_SHADOW_STRENGTH,
            max_size=DEFAULT_MAX_SIZE,
            jpeg_quality=DEFAULT_JPEG_QUALITY,
        ),
    }

@app.post("/compose")
async def compose_api(
    selfie: UploadFile = File(..., description="User selfie (jpg/png)"),
    postcard: UploadFile = File(..., description="Postcard design (jpg/png)"),
    color: Literal["off","reinhard","hist"] = Query(DEFAULT_COLOR),
    color_strength: float = Query(DEFAULT_COLOR_STRENGTH, ge=0.0, le=1.0),
    shadow_strength: float = Query(DEFAULT_SHADOW_STRENGTH, ge=0.0, le=0.4),
    max_size: int = Query(DEFAULT_MAX_SIZE, ge=600, le=3000),
    jpeg_quality: int = Query(DEFAULT_JPEG_QUALITY, ge=60, le=100),
    # если авто не нашло карточку — можно передать 4 точки вручную (x1,y1,...,x4,y4)
    quad: Optional[str] = Query(None, description="Optional 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4"),
):
    try:
        s_bytes = await selfie.read()
        p_bytes = await postcard.read()

        s = cv2.imdecode(np.frombuffer(s_bytes, np.uint8), cv2.IMREAD_COLOR)
        p = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
        if s is None or p is None:
            raise ValueError("Cannot decode input images")

        manual_quad = None
        if quad:
            parts = [float(t) for t in quad.split(",")]
            if len(parts) != 8:
                raise ValueError("quad must have 8 comma-separated numbers")
            manual_quad = np.array(parts, dtype=np.float32).reshape(4,2)

        out = compose_postcard_on_selfie(
            selfie_bgr=s,
            postcard_bgr=p,
            manual_quad=manual_quad,
            max_side=max_size,
            color_mode=color,
            color_alpha=color_strength,
            shadow_alpha=shadow_strength,
        )

        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return StreamingResponse(io.BytesIO(enc.tobytes()), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
