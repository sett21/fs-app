from dataclasses import dataclass
import os

from typing import Optional, Tuple
import cv2, numpy as np
import mediapipe as mp
from PIL import Image

@dataclass
class _HairCfg:
    use_hair_from_src: bool
    band_inner_px: int
    band_outer_px: int
    alpha: float
    poisson_face: bool



def hex_to_bgr(hex_color: str) -> Tuple[int,int,int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[4:6],16), int(hex_color[2:4],16), int(hex_color[0:2],16))

def resize_limit(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    s = max_side / max(h, w)
    new_w, new_h = int(w*s), int(h*s)
    interp = cv2.INTER_AREA if (new_w < w and new_h < h) else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp), s

def skin_mask(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, np.array([0,133,77],np.uint8), np.array([255,173,127],np.uint8))
    m = cv2.medianBlur(m,5)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)

def order_pts(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], np.float32)

def warp_into_quad(src_bgr: np.ndarray, dst_quad: np.ndarray, canvas_size: Tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
    Hc, Wc = canvas_size
    dst_quad = order_pts(dst_quad)
    tw = max(int(np.linalg.norm(dst_quad[1]-dst_quad[0])), 20)
    th = max(int(np.linalg.norm(dst_quad[3]-dst_quad[0])), 20)
    interp_card = cv2.INTER_AREA if (tw < src_bgr.shape[1] and th < src_bgr.shape[0]) else cv2.INTER_LINEAR
    card = cv2.resize(src_bgr, (tw, th), interpolation=interp_card)
    card_pad = cv2.copyMakeBorder(card, 1,1,1,1, cv2.BORDER_REPLICATE)
    src_pts = np.array([[1,1],[tw,1],[tw,th],[1,th]], np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_quad)
    overlay = cv2.warpPerspective(card_pad, M, (Wc, Hc), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((Hc, Wc), np.uint8)
    cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)
    return overlay, mask

def pil_from_bgr(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def bgr_from_pil(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def detect_green_quad(bgr: np.ndarray, hex_color="#00ff84") -> Optional[np.ndarray]:
    Ht = int(cv2.cvtColor(np.uint8([[list(hex_to_bgr(hex_color))]]), cv2.COLOR_BGR2HSV)[0,0,0])
    hshift = int(os.getenv("GREEN_H_SHIFT","15"))
    smin   = int(os.getenv("GREEN_S_MIN","80"))
    vmin   = int(os.getenv("GREEN_V_MIN","60"))
    small, s = resize_limit(bgr, 1600)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower = np.array([max(Ht-hshift,0), smin, vmin], np.uint8)
    upper = np.array([min(Ht+hshift,179), 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), 2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    quad = (approx.reshape(-1,2) if len(approx)==4 else cv2.boxPoints(cv2.minAreaRect(c))).astype(np.float32)
    return quad / s

def green_mask_hsv(bgr: np.ndarray, base_hex="#00ff84",
                   hshift: int=None, smin: int=None, vmin: int=None) -> np.ndarray:
    if hshift is None: hshift = int(os.getenv("GREEN_H_SHIFT","15"))
    if smin   is None: smin   = int(os.getenv("GREEN_S_MIN","80"))
    if vmin   is None: vmin   = int(os.getenv("GREEN_V_MIN","60"))
    Ht = int(cv2.cvtColor(np.uint8([[list(hex_to_bgr(base_hex))]]), cv2.COLOR_BGR2HSV)[0,0,0])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([max(Ht-hshift,0), smin, vmin], np.uint8)
    upper = np.array([min(Ht+hshift,179), 255, 255], np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    return cv2.medianBlur(m, 3)

# ===== Специфика вставки: дефриндж, бленды, детали =====
def green_defringe(base_bgr: np.ndarray, overlay_bgr: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    if edge_mask.ndim == 3:
        edge_mask = edge_mask[..., 0]
    band = cv2.GaussianBlur((edge_mask > 0).astype(np.uint8) * 255, (0, 0), 1.0)
    if band.max() == 0:
        return base_bgr
    b, g, r = cv2.split(base_bgr)
    greenish = ((g.astype(np.int16) - r.astype(np.int16) > 18) &
                (g.astype(np.int16) - b.astype(np.int16) > 18) &
                (g > 120)).astype(np.uint8)
    band = cv2.bitwise_and((band > 0).astype(np.uint8), greenish)
    if band.max() == 0:
        return base_bgr
    g2 = g.copy()
    idx = band.astype(bool)
    g2[idx] = (g2[idx].astype(np.float32) * 0.8).astype(np.uint8)
    fixed = cv2.merge([b, g2, r])
    a = (cv2.GaussianBlur(band * 255, (0, 0), 1.0).astype(np.float32) / 255.0)[:, :, None]
    return (fixed.astype(np.float32) * (1 - a) + overlay_bgr.astype(np.float32) * a).astype(np.uint8)

def srgb_to_lin(x):
    x = x.astype(np.float32)/255.0
    return np.where(x<=0.04045, x/12.92, ((x+0.055)/1.055)**2.4)
def lin_to_srgb(y):
    x = np.where(y<=0.0031308, 12.92*y, 1.055*np.power(y, 1/2.4)-0.055)
    return np.clip(np.round(x*255.0),0,255).astype(np.uint8)

def laplacian_blend(A, B, mask, levels=4):
    A = A.astype(np.float32); B = B.astype(np.float32); M = mask.astype(np.float32)
    gpA=[A]; gpB=[B]; gpM=[M]
    for _ in range(levels-1):
        if min(gpA[-1].shape[:2]) <= 2: break
        gpA.append(cv2.pyrDown(gpA[-1]))
        gpB.append(cv2.pyrDown(gpB[-1]))
        gpM.append(cv2.pyrDown(gpM[-1]))
    lpA=[gpA[-1]]; lpB=[gpB[-1]]
    for i in range(len(gpA)-1,0,-1):
        size=(gpA[i-1].shape[1], gpA[i-1].shape[0])
        upA = cv2.pyrUp(gpA[i], dstsize=size) if min(size) > 2 else gpA[i]
        upB = cv2.pyrUp(gpB[i], dstsize=size) if min(size) > 2 else gpB[i]
        la = gpA[i-1] - upA
        lb = gpB[i-1] - upB
        lpA.append(la); lpB.append(lb)
    LS=[]
    for la,lb,gm in zip(lpA, lpB, gpM[::-1]):
        gm3 = gm if gm.ndim==3 else gm[:,:,None]
        LS.append(la*gm3 + lb*(1-gm3))
    out = LS[0]
    for i in range(1,len(LS)):
        size=(LS[i].shape[1], LS[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + LS[i]
    return np.clip(out,0,255).astype(np.uint8)

def ring_grad(mask_poly: np.ndarray, inner_px=8, outer_px=8):
    inner = cv2.erode(mask_poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(inner_px*2+1,inner_px*2+1)),1)
    outer = cv2.dilate(mask_poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(outer_px*2+1,outer_px*2+1)),1)
    ring = cv2.subtract(outer, inner)
    dist = cv2.distanceTransform(ring, cv2.DIST_L2, 3)
    if dist.max()>0: dist = dist/dist.max()
    return dist  # 0..1

def apply_page_stack(card_bgr: np.ndarray, side="right", inner_px=16, shade=0.06, stripes=True) -> np.ndarray:
    h,w = card_bgr.shape[:2]
    ramp = np.zeros((h,w), np.float32)
    if side=="left":
        ramp[:, :inner_px] = (1 - (np.linspace(0,1,inner_px)[None,:]))
    elif side=="right":
        ramp[:, w-inner_px:] = (np.linspace(0,1,inner_px)[None,:])
    elif side=="top":
        ramp[:inner_px, :] = (1 - (np.linspace(0,1,inner_px)[:,None]))
    else:
        ramp[h-inner_px:, :] = (np.linspace(0,1,inner_px)[:,None])
    ramp = cv2.GaussianBlur(ramp, (0,0), 1.0)
    mult = 1.0 - shade*ramp
    out = (card_bgr.astype(np.float32) * mult[:,:,None]).clip(0,255).astype(np.uint8)
    if stripes and inner_px>=8:
        step = 4; a = 0.12
        for i in range(2, inner_px-2, step):
            if side in ("left","right"):
                x = i if side=="left" else w-1-i
                out[:, x:x+1] = (out[:, x:x+1].astype(np.float32)*(1-a)).astype(np.uint8)
            else:
                y = i if side=="top" else h-1-i
                out[y:y+1, :] = (out[y:y+1, :].astype(np.float32)*(1-a)).astype(np.uint8)
    return out

def _safe_seamless_clone(src, dst, mask, center):
    if mask is None or mask.max() == 0: return dst
    try: return cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)
    except Exception: return dst

# ===== Доп. улучшалки качества =====
def highpass_detail(img: np.ndarray, sigma=0.9, gain=0.22) -> np.ndarray:
    low = cv2.GaussianBlur(img, (0,0), sigma)
    hp  = cv2.addWeighted(img, 1.0, low, -1.0, 0)
    out = cv2.addWeighted(img, 1.0, hp, gain, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def refine_alpha_with_edges(bgr: np.ndarray, a: np.ndarray, iters: int=1) -> np.ndarray:
    # a: HxW в [0,1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1).astype(np.float32)/255.0
    aa = a.copy().astype(np.float32)
    for _ in range(max(1, iters)):
        aa = np.clip(aa + 0.25*edges*(1.0 - 2.0*np.abs(aa-0.5)), 0, 1)
    return aa

def texture_reinject(dst_bgr: np.ndarray, src_bgr: np.ndarray, mask01: np.ndarray, sigma=1.1, gain=0.35) -> np.ndarray:
    # mask01 HxW [0,1]
    src_low = cv2.GaussianBlur(src_bgr, (0,0), sigma)
    dst_low = cv2.GaussianBlur(dst_bgr, (0,0), sigma)
    src_hp  = cv2.addWeighted(src_bgr, 1.0, src_low, -1.0, 0)
    dst_hp  = cv2.addWeighted(dst_bgr, 1.0, dst_low, -1.0, 0)
    mix_hp  = np.clip(dst_hp + gain*src_hp, -255, 255).astype(np.float32)
    m3 = mask01[:,:,None].astype(np.float32)
    out = np.clip(dst_bgr.astype(np.float32) + m3*mix_hp, 0, 255).astype(np.uint8)
    return out

def add_fine_grain(bgr: np.ndarray, strength=0.012) -> np.ndarray:
    if strength <= 0: return bgr
    h,w = bgr.shape[:2]
    noise = (np.random.randn(h,w,1).astype(np.float32) * 255.0).clip(-255,255)
    return np.clip(bgr.astype(np.float32) + strength*noise, 0, 255).astype(np.uint8)

def bbox_from_mask(mask: np.ndarray, pad: int=8) -> Tuple[int,int,int,int]:
    ys, xs = np.where(mask>0)
    if xs.size == 0: return 0,0,mask.shape[1],mask.shape[0]
    x0,x1 = xs.min(), xs.max()
    y0,y1 = ys.min(), ys.max()
    x0 = max(0, x0-pad); y0 = max(0, y0-pad)
    x1 = min(mask.shape[1], x1+pad+1); y1 = min(mask.shape[0], y1+pad+1)
    return x0,y0,x1,y1

def build_contact_masks_tight(hand_mask: np.ndarray, poly_mask: np.ndarray):
    # узкий контактный «шов» по пересечению руки и кромки документа + узкий пояс вокруг полигона
    edge = cv2.morphologyEx(poly_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    contact = cv2.bitwise_and(hand_mask, edge)
    contact = cv2.dilate(contact, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    narrow = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return contact, narrow



# ===== MediaPipe Hands (ленивая инициализация) =====
_mp_hands = None
def get_mp_hands():
    if os.getenv("DISABLE_HANDS", "0") == "1":
        return None
    global _mp_hands
    if _mp_hands is None:
        _mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    return _mp_hands

def hand_mask_bgr(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if os.getenv("DISABLE_HANDS", "0") == "1":
        return np.zeros((h, w), np.uint8)
    hands = get_mp_hands()
    if hands is None:
        return np.zeros((h, w), np.uint8)
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
    except Exception:
        return np.zeros((h, w), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    if not res or not res.multi_hand_landmarks:
        return mask
    for hand in res.multi_hand_landmarks:
        pts = np.array([[int(lm.x*w), int(lm.y*h)] for lm in hand.landmark], np.int32)
        if pts.shape[0] >= 3:
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
    # Усиление маски за счёт кожи (безопасно, т.к. позже пересекается с полигоном)
    try:
        if os.getenv("HAND_SKIN_UNION", "1") == "1":
            skin = skin_mask(bgr)
            mask = cv2.bitwise_or(mask, skin)
    except Exception:
        pass
    dil = max(3, int(os.getenv("HAND_DILATE_PX", "6")))
    ero = max(0, int(os.getenv("HAND_ERODE_PX", "2")))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil*2+1, dil*2+1))
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero*2+1, ero*2+1)) if ero>0 else None
    mask = cv2.dilate(mask, k_d, 1)
    if k_e is not None:
        mask = cv2.erode(mask, k_e, 1)
    return mask
