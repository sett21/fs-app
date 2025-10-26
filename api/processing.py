import cv2, numpy as np
from typing import Optional, Tuple

# ---------- helpers ----------

def _resize_limit(img, max_side=1600):
    h, w = img.shape[:2]
    if max(h, w) <= max_side: return img, 1.0
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), cv2.INTER_AREA), scale

def _order_pts(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],        # TL
        pts[np.argmin(diff)],     # TR
        pts[np.argmax(s)],        # BR
        pts[np.argmax(diff)],     # BL
    ], dtype=np.float32)

def _warp_quad(src, quad, size: Tuple[int,int]):
    quad = _order_pts(quad)
    W, H = size
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(src, M, (W,H))

# ---------- color ----------

def reinhard_color_transfer(src_bgr, ref_bgr):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        s_mean, s_std = src[:,:,c].mean(), src[:,:,c].std() + 1e-6
        r_mean, r_std = ref[:,:,c].mean(), ref[:,:,c].std() + 1e-6
        src[:,:,c] = (src[:,:,c] - s_mean) * (r_std / s_std) + r_mean
    src = np.clip(src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src, cv2.COLOR_LAB2BGR)

def _hist_match_channel(src, ref):
    s_values, bin_idx, s_counts = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(ref.ravel(), return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / src.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / ref.size
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
    return interp_r_values[bin_idx].reshape(src.shape).astype(np.uint8)

def histogram_match_bgr(src_bgr, ref_bgr):
    return cv2.merge([_hist_match_channel(src_bgr[:,:,i], ref_bgr[:,:,i]) for i in range(3)])

def _blend(a, b, alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(b, alpha, a, 1.0 - alpha, 0)

# ---------- detection: find card/phone/paper quad ----------

def detect_card_quad(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Ищем наиболее «картоподобный» четырёхугольник: яркая/равномерная область,
    ровные грани. Возвращает 4 точки (float32) в координатах исходника или None.
    """
    small, scale = _resize_limit(img, 1400)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    v = cv2.bilateralFilter(v, 7, 50, 50)
    edges = cv2.Canny(v, 40, 120)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    area_img = small.shape[0]*small.shape[1]

    for c in cnts:
        if cv2.contourArea(c) < 0.002 * area_img:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(approx)
        (rw, rh) = rect[1]
        if rw < 10 or rh < 10:
            continue
        rectangularity = cv2.contourArea(approx) / (rw*rh + 1e-6)
        a = max(rw, rh) / max(min(rw, rh), 1e-6)
        if rectangularity < 0.7 or a > 2.5:
            continue
        score = rectangularity * (cv2.contourArea(approx) / area_img)
        if score > best_score:
            best_score = score
            best = approx.reshape(4,2).astype(np.float32) / scale

    return best

# ---------- composition ----------

def compose_postcard_on_selfie(
    selfie_bgr: np.ndarray,
    postcard_bgr: np.ndarray,
    manual_quad: Optional[np.ndarray] = None,
    max_side: int = 1600,
    color_mode: str = "reinhard",
    color_alpha: float = 0.8,
    shadow_alpha: float = 0.15,
) -> np.ndarray:
    # 1) ограничим размер селфи
    selfie, scale0 = _resize_limit(selfie_bgr, max_side)
    H, W = selfie.shape[:2]

    # 2) найти/принять четырёхугольник карты
    quad = manual_quad if manual_quad is not None else detect_card_quad(selfie)
    if quad is None:
        # fallback: центрированный прямоуг.
        w = int(W * 0.45)
        h = int(w * 0.7)
        x0 = (W - w)//2
        y0 = int(H*0.45)
        quad = np.array([[x0,y0],[x0+w,y0],[x0+w,y0+h],[x0,y0+h]], np.float32)

    # 3) размеры целевой области
    wA = np.linalg.norm(quad[1]-quad[0]); wB = np.linalg.norm(quad[2]-quad[3])
    hA = np.linalg.norm(quad[3]-quad[0]); hB = np.linalg.norm(quad[2]-quad[1])
    tw = max(int(max(wA,wB)), 20)
    th = max(int(max(hA,hB)), 20)

    # 4) подготовка открытки
    card_src = cv2.resize(postcard_bgr, (tw, th), cv2.INTER_AREA)

    # 5) цветокоррекция открытки под локальный фон (по желанию)
    if color_mode != "off":
        x0, y0 = quad[:,0].min(), quad[:,1].min()
        x1, y1 = quad[:,0].max(), quad[:,1].max()
        pad = int(0.1*max(tw,th))
        xa, ya = max(int(x0-pad),0), max(int(y0-pad),0)
        xb, yb = min(int(x1+pad),W-1), min(int(y1+pad),H-1)
        ref = selfie[ya:yb+1, xa:xb+1].copy()
        ref = cv2.resize(ref, (tw, th), cv2.INTER_AREA)
        if color_mode == "reinhard":
            corr = reinhard_color_transfer(card_src, ref)
        else:
            corr = histogram_match_bgr(card_src, ref)
        card_src = _blend(card_src, corr, color_alpha)

    # 6) перспективная вклейка в селфи
    dst_quad = _order_pts(quad)
    src_pts = np.array([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]], np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_quad)
    overlay = cv2.warpPerspective(card_src, M, (W, H))

    # 7) маска области + мягкая тень по краю
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (0,0), 1.2)

    if shadow_alpha > 0:
        edge = cv2.GaussianBlur(mask, (0,0), 2.5)
        edge = cv2.normalize(edge.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        shadow = (1.0 - shadow_alpha * edge)[..., None]
        base = (selfie.astype(np.float32) * shadow).astype(np.uint8)
    else:
        base = selfie.copy()

    m = (mask.astype(np.float32)/255.0)[...,None]
    out = (overlay.astype(np.float32)*m + base.astype(np.float32)*(1-m)).astype(np.uint8)
    return out
