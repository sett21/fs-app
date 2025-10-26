import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # запрет Transformers импортировать torchvision
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import cv2, numpy as np, torch
from typing import Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import mediapipe as mp
import insightface
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline

# ===== CPU-friendly =====
torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "2"))))
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))

_FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
              152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]

@dataclass
class _HairCfg:
    use_hair_from_src: bool
    band_inner_px: int
    band_outer_px: int
    alpha: float          # максимум прозрачности источника в смеси (0..1)
    poisson_face: bool

# ===== MediaPipe Hands =====
_mp_hands = None

def get_mp_hands():
    """Ленивая инициализация Mediapipe Hands с учётом флага выключения."""
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
    """Возвращает бинарную маску рук (0/255). Никогда не возвращает None."""
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
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)

# ===== Utils =====
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

def build_edge_mask_binary(selfie_bgr: np.ndarray, quad: np.ndarray, H: int, W: int,
                           inner_px: int, outer_px: int) -> np.ndarray:
    """Строго бинарная (0/255) узкая корона вокруг кромки + кожа рядом."""
    poly = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(poly, quad.astype(np.int32), 255)

    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_px*2+1, inner_px*2+1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_px*2+1, outer_px*2+1))
    inner_erode  = cv2.erode(poly, k_in, 1)
    ring_inside  = cv2.subtract(poly, inner_erode)
    outer_dilate = cv2.dilate(poly, k_out, 1)
    ring_outside = cv2.subtract(outer_dilate, poly)
    ring = cv2.bitwise_or(ring_inside, ring_outside)

    near_edge = cv2.dilate(poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    ring = cv2.bitwise_or(ring, cv2.bitwise_and(near_edge, skin_mask(selfie_bgr)))

    _, ring_bin = cv2.threshold(ring, 127, 255, cv2.THRESH_BINARY)
    return ring_bin

def order_pts(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], np.float32)

def warp_into_quad(src_bgr: np.ndarray, dst_quad: np.ndarray, canvas_size: Tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
    """Перспективная вклейка src в четырехугольник dst_quad на холсте (H, W)."""
    Hc, Wc = canvas_size
    dst_quad = order_pts(dst_quad)
    tw = max(int(np.linalg.norm(dst_quad[1]-dst_quad[0])), 20)
    th = max(int(np.linalg.norm(dst_quad[3]-dst_quad[0])), 20)

    interp_card = cv2.INTER_AREA if (tw < src_bgr.shape[1] and th < src_bgr.shape[0]) else cv2.INTER_LINEAR
    card = cv2.resize(src_bgr, (tw, th), interpolation=interp_card)
    card_pad = cv2.copyMakeBorder(card, 1,1,1,1, cv2.BORDER_REPLICATE)
    src_pts = np.array([[1,1],[tw,1],[tw,th],[1,th]], np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_quad)
    overlay = cv2.warpPerspective(card_pad, M, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    mask = np.zeros((Hc, Wc), np.uint8)
    cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)
    return overlay, mask

def pil_from_bgr(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def bgr_from_pil(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def detect_green_quad(bgr: np.ndarray, hex_color="#00ff84") -> Optional[np.ndarray]:
    """Поиск зелёного прямоугольника (#00ff84). Возвращает 4 точки в координатах исходника."""
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

# ===== локальный дефриндж «зелени» =====
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

# ===== линейный свет и Laplacian blend по узкому кольцу =====
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

# ===== лёгкий «стопка страниц» =====
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

# ===== безопасный Poisson =====
def _safe_seamless_clone(src, dst, mask, center):
    if mask is None or mask.max() == 0: return dst
    try: return cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)
    except Exception: return dst

# ===== Inpainter =====
class Inpainter:
    def __init__(self, model_repo: str, torch_dtype: str="fp16", device: str="cuda"):
        use_cpu = os.getenv("USE_CPU","0") == "1"
        device = "cpu" if use_cpu else device
        dtype  = torch.float32 if (device=="cpu" or torch_dtype.lower()!="fp16") else torch.float16
        repo_l = model_repo.lower()
        is_sdxl = ("xl" in repo_l) or ("sdxl" in repo_l)
        pipe_cls = StableDiffusionXLInpaintPipeline if is_sdxl else StableDiffusionInpaintPipeline
        force_bin = "runwayml/stable-diffusion-inpainting" in repo_l
        def _load(use_st):
            return pipe_cls.from_pretrained(
                model_repo, torch_dtype=dtype,
                safety_checker=None, feature_extractor=None,
                low_cpu_mem_usage=True, use_safetensors=(use_st and not force_bin)
            )
        try: self.pipe = _load(True)
        except Exception: self.pipe = _load(False)
        try: self.pipe.enable_attention_slicing()
        except Exception: pass
        try: self.pipe.enable_vae_tiling()
        except Exception: pass
        if device != "cpu":
            try: self.pipe.enable_sequential_cpu_offload()
            except Exception: pass
            self.pipe = self.pipe.to(device)
        self.device = device

    @torch.inference_mode()
    def inpaint(self, init_rgb: Image.Image, mask_rgb: Image.Image,
                prompt: Optional[str]=None, negative: Optional[str]=None,
                steps: int=24, guidance: float=5.0, strength: float=0.5) -> Image.Image:
        if not prompt or not str(prompt).strip():
            prompt = os.getenv("DEFAULT_PROMPT",
                "photorealistic selfie; person holding a postcard; "
                "keep postcard exactly unchanged; soft contact shadows; realistic fingers")
        if negative is None:
            negative = os.getenv("NEG_PROMPT",
                "altered postcard, changed text, boosted contrast, hdr halos, "
                "deformed hands, extra fingers, warped text, logo distortion, blur, artifacts")
        if mask_rgb.mode != "L":
            mask_rgb = mask_rgb.convert("L")
        return self.pipe(
            prompt=prompt, negative_prompt=negative,
            image=init_rgb, mask_image=mask_rgb,
            guidance_scale=guidance, num_inference_steps=steps, strength=strength
        ).images[0]

# ===== Face Swap =====
class FaceSwapper:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face model not found: {model_path}")
        home = os.getenv("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))
        os.makedirs(os.path.join(home, "models"), exist_ok=True)

        self.app = insightface.app.FaceAnalysis(name="buffalo_l", root=home)
        ctx = -1 if os.getenv("USE_CPU","0") == "1" else 0
        det = int(os.getenv("INSIGHTFACE_DET_SIZE", "640"))
        self.app.prepare(ctx_id=ctx, det_size=(det, det))
        providers = ['CPUExecutionProvider'] if ctx == -1 else ['CUDAExecutionProvider','CPUExecutionProvider']
        self.swapper = insightface.model_zoo.get_model(model_path, providers=providers)

        # facemesh по флагу
        if os.getenv("DISABLE_FACEMESH", "0") == "1":
            self.fm = None
        else:
            try:
                self.fm = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=1,
                    min_detection_confidence=0.4
                )
            except Exception:
                self.fm = None

        self.cfg = _HairCfg(
            use_hair_from_src = os.getenv("HAIR_FROM_SRC","0") == "1",
            band_inner_px     = int(os.getenv("HAIR_BAND_INNER_PX","12")),
            band_outer_px     = int(os.getenv("HAIR_BAND_OUTER_PX","24")),
            alpha             = float(os.getenv("HAIR_ALPHA","0.8")),
            poisson_face      = os.getenv("POISSON_FACE","1") == "1",
        )

    def _facemesh_oval(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.fm is None:
            return None
        h, w = bgr.shape[:2]
        res = self.fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res or not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = np.array([[int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)] for i in _FACE_OVAL], np.int32)
        return pts

    def _oval_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        pts = self._facemesh_oval(bgr)
        if pts is not None:
            cv2.fillConvexPoly(mask, pts, 255)
        return mask

    def _hairline_band_src(self, bgr: np.ndarray, inner_px: int, outer_px: int) -> np.ndarray:
        h, w = bgr.shape[:2]
        oval = self._oval_mask(bgr)
        if oval.max()==0:
            return np.zeros((h,w), np.uint8)
        ys, xs = np.where(oval>0)
        y_min, y_max = ys.min(), ys.max(); y_mid = int(y_min + 0.55*(y_max - y_min))
        upper = np.zeros_like(oval); upper[:y_mid,:] = 255
        inner = cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(inner_px*2+1,inner_px*2+1)),1)
        outer = cv2.dilate(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(outer_px*2+1,outer_px*2+1)),1)
        ring  = cv2.subtract(outer, inner)
        ring  = cv2.bitwise_and(ring, upper)
        ring  = cv2.GaussianBlur(ring, (0,0), 1.2)
        _, ring = cv2.threshold(ring, 16, 255, cv2.THRESH_BINARY)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(oval))  # не лезем внутрь лица
        return ring

    def swap(self, target_bgr: np.ndarray, src_face_bgr: np.ndarray) -> np.ndarray:
        tar_faces = self.app.get(target_bgr)
        src_faces = self.app.get(src_face_bgr)
        if not tar_faces or not src_faces:
            return target_bgr

        tar = tar_faces[0]; src = src_faces[0]
        rough = self.swapper.get(target_bgr.copy(), tar, src, paste_back=True)

        h, w = target_bgr.shape[:2]
        oval = self._oval_mask(rough)
        if oval.max() == 0:
            return rough

        eye_dist = float(np.linalg.norm(src.kps[0] - src.kps[1])) + 1e-6
        scale = max(0.6, min(1.6, eye_dist/55.0))
        inner = max(8, int(self.cfg.band_inner_px * scale))
        outer = max(inner+6, int(self.cfg.band_outer_px * scale))
        hair_band_src = self._hairline_band_src(src_face_bgr, inner, outer)

        M, _ = cv2.estimateAffinePartial2D(src.kps.astype(np.float32),
                                           tar.kps.astype(np.float32),
                                           method=cv2.LMEDS)
        hair_band_t = cv2.warpAffine(hair_band_src, M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT)

        ear_cut = cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)), 1)
        face_core = cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
        hair_band_t = cv2.bitwise_and(hair_band_t, cv2.bitwise_not(ear_cut))
        mix_mask = cv2.bitwise_or(face_core, hair_band_t)     # 0/255
        mix_alpha = cv2.GaussianBlur(mix_mask, (0,0), 1.6).astype(np.float32)/255.0
        mix_alpha = mix_alpha[:,:,None]

        # Reinhard color match внутри маски
        def _reinhard_to_ref(src_img, ref_img, msk):
            m = msk > 0
            if not np.any(m): return src_img
            A = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            B = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            for c in range(3):
                mu_a, sd_a = A[...,c][m].mean(), A[...,c][m].std()+1e-6
                mu_b, sd_b = B[...,c][m].mean(), B[...,c][m].std()+1e-6
                A[...,c][m] = (A[...,c][m]-mu_a)*(sd_b/sd_a) + mu_b
            return cv2.cvtColor(np.clip(A,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        rough_matched = _reinhard_to_ref(rough, target_bgr, mix_mask)

        # Сохраняем очки/жёсткие границы из таргета
        gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
        glasses_mask = cv2.bitwise_and(edges, oval)
        glasses_mask = cv2.GaussianBlur(glasses_mask, (0,0), 1.2)
        glasses_a = (glasses_mask.astype(np.float32)/255.0)[:,:,None] * 0.85

        out = (rough_matched.astype(np.float32)*mix_alpha + target_bgr.astype(np.float32)*(1-mix_alpha)).astype(np.uint8)
        if glasses_a.max() > 0:
            out = (target_bgr.astype(np.float32)*glasses_a + out.astype(np.float32)*(1-glasses_a)).astype(np.uint8)

        # Узкий Poisson по периметру
        band = cv2.morphologyEx(mix_mask, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        ys, xs = np.where(mix_mask>0)
        if len(xs) > 0:
            center = (int(xs.mean()), int(ys.mean()))
            out = _safe_seamless_clone(out, target_bgr, band, center)

        return out


# ===== Основной пайплайн =====
class GenPipeline:
    def __init__(self):
        # --- Inpaint / SD ---
        self.model_repo  = os.getenv("INPAINT_REPO", "runwayml/stable-diffusion-inpainting")
        self.dtype       = os.getenv("TORCH_dtype", "fp32")
        self.steps       = int(os.getenv("STEPS", "18"))
        self.guidance    = float(os.getenv("GUIDANCE", "5.0"))
        self.strength    = float(os.getenv("STRENGTH", "0.5"))
        self.neg_prompt  = os.getenv("NEG_PROMPT",
            "altered postcard, changed text, boosted contrast, glow edges, hdr halos, "
            "extra fingers, deformed hands, blur, artifacts, warped text, logo distortion")

        # --- поведение по краям / перья ---
        self.ring_in_pct  = float(os.getenv("RING_IN_PCT",  "0.004"))
        self.ring_out_pct = float(os.getenv("RING_OUT_PCT", "0.012"))
        self.seamless_edge = os.getenv("SEAMLESS_EDGE", "1") == "1"

        # --- эффекты страницы ---
        self.page_effect   = os.getenv("PAGE_EFFECT","1") == "1"
        self.page_side     = os.getenv("PAGE_SIDE","right")
        self.page_inner_px = int(os.getenv("PAGE_INNER_PX","16"))
        self.page_shade    = float(os.getenv("PAGE_SHADE","0.06"))
        self.page_stripes  = os.getenv("PAGE_STRIPES","1") == "1"

        # --- swap режим ---
        self.disable_swap = os.getenv("DISABLE_FACE_SWAP","0") == "1"
        swap_mode_env = os.getenv("SWAP_MODE","").lower().strip()
        if not swap_mode_env:
            swap_mode_env = "before" if os.getenv("SWAP_BEFORE_COMPOSE","1") == "1" else "after"
        self.swap_mode = swap_mode_env  # "before" | "after"

        # --- инициализация подсистем ---
        self.inpainter   = Inpainter(self.model_repo, self.dtype)
        self.swapper     = FaceSwapper(os.getenv("FACE_MODEL_PATH", "/models/inswapper_128.onnx"))

    def run(self, selfie_bgr: np.ndarray, postcard_bgr: np.ndarray, face_bgr: np.ndarray,
            prompt: Optional[str]=None) -> np.ndarray:
        # ===== 0) Препроцесс =====
        if selfie_bgr is None or postcard_bgr is None:
            raise ValueError("Empty input: selfie or postcard is None")

        selfie_bgr, _ = resize_limit(selfie_bgr, MAX_SIDE)
        face_bgr,   _ = resize_limit(face_bgr,   MAX_SIDE)
        if os.getenv("KEEP_POSTCARD_RES", "1") != "1":
            postcard_bgr, _ = resize_limit(postcard_bgr, MAX_SIDE)

        H, W = selfie_bgr.shape[:2]

        # ===== 1) Faceswap (по режиму) =====
        if not self.disable_swap and self.swap_mode == "before":
            print("[faceswap] mode=before", flush=True)
            try:
                selfie_bgr = self.swapper.swap(selfie_bgr, face_bgr)
            except Exception as e:
                print(f"[faceswap][before] failed: {e}", flush=True)

        # ===== 2) Поиск зелёного прямоугольника =====
        quad = detect_green_quad(selfie_bgr, "#00ff84")
        if quad is None or quad.shape != (4,2):
            raise RuntimeError("Green rectangle (#00ff84) not found or invalid")

        # ===== 3) Нежный «книжный» край (не трогаем содержимое) =====
        if self.page_effect:
            try:
                postcard_bgr = apply_page_stack(
                    postcard_bgr, side=self.page_side,
                    inner_px=max(8, self.page_inner_px),
                    shade=self.page_shade,
                    stripes=self.page_stripes
                )
            except Exception as e:
                print(f"[page_effect] skipped: {e}", flush=True)

        # ===== 4) Перспективная вклейка открытки в кадр =====
        overlay, mask_poly = warp_into_quad(postcard_bgr, quad, (H, W))

        # ===== 5) HAND-AWARE размещение: открытка позади руки =====
        hand   = hand_mask_bgr(selfie_bgr)            # 255 = рука
        nohand = (hand == 0).astype(np.uint8)
        place = (nohand & (mask_poly>0).astype(np.uint8)).astype(np.float32)  # 0/1
        alpha_hard = place[:, :, None]  # 0..1 без пера внутри документа

        # линейный свет для аккуратного микса
        base_lin = srgb_to_lin(selfie_bgr)
        ov_lin   = srgb_to_lin(overlay)
        init     = lin_to_srgb(ov_lin*alpha_hard + base_lin*(1-alpha_hard))

        # ===== 6) Скрыть шов: тонкий laplacian-blend только в узком поясе =====
        rin  = max(1, int(self.ring_in_pct  * min(H, W)))
        rout = max(3, int(self.ring_out_pct * min(H, W)))
        edge_grad  = ring_grad(mask_poly, rin, rout)                 # 0..1
        alpha_edge = (edge_grad[:, :, None] * (nohand[:, :, None]/255.0)).astype(np.float32)
        init = laplacian_blend(overlay, init, alpha_edge)

        # ===== 7) Дефриндж зелени (край + под пальцами) =====
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        edge_band  = cv2.morphologyEx(mask_poly, cv2.MORPH_GRADIENT, k3)
        under_hand = cv2.bitwise_and(hand, mask_poly)
        to_despill = cv2.bitwise_or(edge_band, under_hand)
        init = green_defringe(init, overlay, to_despill)

        # ===== 8) Узкая маска для inpaint (пальцы/тени) =====
        ring_base   = build_edge_mask_binary(selfie_bgr, quad, H, W, rin, max(rout, rin+4))
        hand_erode  = cv2.erode(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)
        ring_wo_hand= cv2.bitwise_and(ring_base, cv2.bitwise_not(hand_erode))
        contact_band= cv2.dilate(cv2.bitwise_and(ring_base, hand),
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
        mask_edge   = cv2.bitwise_or(ring_wo_hand, contact_band)
        _, mask_edge= cv2.threshold(mask_edge, 127, 255, cv2.THRESH_BINARY)

        # (опц.) лёгкая компенсация света по кромке
        if self.seamless_edge:
            ys, xs = np.where(mask_poly > 0)
            if len(xs) > 0:
                center = (int(xs.mean()), int(ys.mean()))
                try:
                    init = cv2.seamlessClone(overlay, init, edge_band, center, cv2.MIXED_CLONE)
                except Exception as e:
                    print(f"[seamless_edge] skipped: {e}", flush=True)

        # ===== 8.5) Полный скип SD-inpaint по флагу (надёжный режим) =====
        if os.getenv("SKIP_INPAINT", "0") == "1":
            final_bgr = init
            swap_mode = (os.getenv("SWAP_MODE", "").lower()
                         or ("before" if os.getenv("SWAP_BEFORE_COMPOSE","1")=="1" else "after"))
            do_swap   = os.getenv("DISABLE_FACE_SWAP","0") != "1"
            if do_swap and swap_mode == "after":
                print("[faceswap] mode=after", flush=True)
                try:
                    final_bgr = self.swapper.swap(final_bgr, face_bgr)
                except Exception as e:
                    print(f"[faceswap][after][skip_inpaint] failed: {e}", flush=True)
            return final_bgr

        # ===== 8.7) Лёгкий CPU-fallback (без diffusers) =====
        use_cpu = os.getenv("USE_CPU","0") == "1"
        force_cv = os.getenv("FALLBACK_INPAINT","").lower() in {"1","cv"}
        if use_cpu or force_cv:
            dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            small_mask = cv2.dilate(mask_edge, dil, 1)
            cv_mode = cv2.INPAINT_NS  # или cv2.INPAINT_TELEA
            inpainted = cv2.inpaint(init, small_mask, 3, cv_mode)
            final_bgr = inpainted
            if not self.disable_swap and self.swap_mode == "after":
                print("[faceswap] mode=after", flush=True)
                try:
                    final_bgr = self.swapper.swap(final_bgr, face_bgr)
                except Exception as e:
                    print(f"[faceswap][after][cv_fallback] {e}", flush=True)
            return final_bgr

        # ===== 9) SD Inpaint (c даунскейлом, чтобы не ловить OOM) =====
        _r8 = lambda x: (x+7)//8*8
        W8, H8   = _r8(W), _r8(H)
        init_8   = cv2.resize(init, (W8, H8), interpolation=cv2.INTER_LINEAR)
        mask_8   = cv2.resize(mask_edge, (W8, H8), interpolation=cv2.INTER_NEAREST)

        limit = int(os.getenv("INPAINT_SIDE_LIMIT","0"))  # 0 = без даунскейла
        if limit and limit > 0:
            scale = min(1.0, float(limit) / max(W8, H8))
            w_s, h_s = max(64, int(W8*scale)), max(64, int(H8*scale))
            init_s = cv2.resize(init_8, (w_s, h_s), interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize(mask_8, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
            steps = max(6, self.steps//2)
            guidance = min(self.guidance, 4.0)
            strength = min(self.strength, 0.45)
            in_img = init_s; in_mask = mask_s
        else:
            steps = self.steps; guidance = self.guidance; strength = self.strength
            in_img = init_8; in_mask = mask_8

        out_img = self.inpainter.inpaint(
            init_rgb=pil_from_bgr(in_img),
            mask_rgb=Image.fromarray(in_mask),
            prompt=prompt,
            negative=self.neg_prompt,
            steps=steps, guidance=guidance, strength=strength
        )

        out_bgr_s = bgr_from_pil(out_img)
        out_bgr = cv2.resize(out_bgr_s, (W, H), interpolation=cv2.INTER_LINEAR)

        # жёсткий композит по узкой маске (всё вне неё — из init)
        m = (mask_edge.astype(np.float32)/255.0)[:, :, None]
        final_bgr = (out_bgr.astype(np.float32)*m + init.astype(np.float32)*(1-m)).astype(np.uint8)

        # ===== 10) Faceswap (после), если так выбран режим =====
        if not self.disable_swap and self.swap_mode == "after":
            print("[faceswap] mode=after", flush=True)
            try:
                final_bgr = self.swapper.swap(final_bgr, face_bgr)
            except Exception as e:
                print(f"[faceswap][after] failed: {e}", flush=True)

        # ===== 11) Отладка по флагу =====
        if os.getenv("SAVE_DEBUG","0") == "1":
            try:
                cv2.imwrite("/tmp/debug_mask_poly.png", mask_poly)
                cv2.imwrite("/tmp/debug_edge_grad.png", (edge_grad*255).astype(np.uint8))
                cv2.imwrite("/tmp/debug_edge_band.png", edge_band)
                cv2.imwrite("/tmp/debug_mask_edge.png", mask_edge)
                cv2.imwrite("/tmp/debug_init.jpg", init)
                cv2.imwrite("/tmp/debug_final.jpg", final_bgr)
            except Exception as e:
                print(f"[debug_save] failed: {e}", flush=True)

        return final_bgr
