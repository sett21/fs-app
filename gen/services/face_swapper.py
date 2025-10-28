import os, cv2, numpy as np, mediapipe as mp
import insightface
from gen.utils.utils import _HairCfg  # оставляем как у тебя

# если _FACE_OVAL не импортируется из utils — раскомментируй и подставь свой список индексов
_FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
              152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
_FACE_CHIN_IDX = 152  # подбородок (FaceMesh)

def _ellipse_kernel(rx, ry):
    rx = max(1, int(rx)); ry = max(1, int(ry))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rx*2+1, ry*2+1))

class FaceSwapper:
   def _rescale_faces(faces, scale):
    """Смасштабировать bbox/kps обратно после инференса на ресайзнутом кадре."""
    if scale == 1.0 or not faces:
        return faces
    out = []
    for f in faces:
        g = type(f)()  # новый объект того же класса
        g.bbox = (f.bbox / scale).astype(f.bbox.dtype)
        g.kps  = (f.kps  / scale).astype(f.kps.dtype) if getattr(f, "kps", None) is not None else None
        g.det_score = getattr(f, "det_score", None)
        out.append(g)
    return out

def _detect_faces(self, img):
    """Ретраи детектора: обычный -> пониже порог -> апскейл -> (опц.) ещё апскейл."""
    faces = self.app.get(img)
    if faces:
        return faces

    # временно понижаем порог
    det_thresh_bak = getattr(self.app, "det_thresh", None)
    try:
        self.app.det_thresh = min(0.25, float(os.getenv("INSIGHTFACE_DET_THRESH", "0.25")))
    except Exception:
        pass

    # апскейл ×1.6
    h, w = img.shape[:2]
    scale = 1.6 if max(h, w) < 1400 else 1.2
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    faces = self.app.get(big)
    if faces:
        try:
            if det_thresh_bak is not None:
                self.app.det_thresh = det_thresh_bak
        except Exception:
            pass
        return _rescale_faces(faces, scale)

    # ещё попытка — лёгкая резкость + CLAHE по яркости
    yuv = cv2.cvtColor(big, cv2.COLOR_BGR2YUV)
    y = yuv[..., 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    yuv[..., 0] = y
    big2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    sharp = cv2.GaussianBlur(big2, (0,0), 1.2)
    big2 = np.clip(big2.astype(np.float32) + 0.6*(big2.astype(np.float32) - sharp.astype(np.float32)), 0, 255).astype(np.uint8)
    faces = self.app.get(big2)

    try:
        if det_thresh_bak is not None:
            self.app.det_thresh = det_thresh_bak
    except Exception:
        pass

    if faces:
        return _rescale_faces(faces, scale)
    return []

def swap(self, target_bgr: np.ndarray, src_face_bgr: np.ndarray, quad_center=None) -> np.ndarray:
    if target_bgr is None or src_face_bgr is None:
        print("[faceswap] empty inputs", flush=True)
        return target_bgr

    tar_faces = self._detect_faces(target_bgr)
    src_faces = self._detect_faces(src_face_bgr)

    print(f"[faceswap] detected target={len(tar_faces)} src={len(src_faces)}", flush=True)

    if not src_faces:
        if os.getenv("SAVE_DEBUG","0") == "1":
            try:
                cv2.imwrite("/tmp/debug_src_no_face.jpg", src_face_bgr)
            except Exception: pass
        print("[faceswap] source faces not found -> skip", flush=True)
        return target_bgr
    if not tar_faces:
        print("[faceswap] target faces not found -> skip", flush=True)
        return target_bgr

    tar = self._pick_target_face(tar_faces, img_shape=target_bgr.shape[:2], prefer_xy=quad_center)
    src = self._pick_target_face(src_faces, img_shape=src_face_bgr.shape[:2], prefer_xy=None)
    if tar is None or src is None:
        print("[faceswap] selection failed", flush=True)
        return target_bgr

    # лог боксов
    tx1, ty1, tx2, ty2 = tar.bbox.astype(int)
    sx1, sy1, sx2, sy2 = src.bbox.astype(int)
    print(f"[faceswap] target bbox=({tx1},{ty1},{tx2},{ty2}) src bbox=({sx1},{sy1},{sx2},{sy2})", flush=True)

    # сам своп
    rough = self.swapper.get(target_bgr.copy(), tar, src, paste_back=True)
    if rough is None:
        print("[faceswap] swapper returned None", flush=True)
        return target_bgr

    diff0 = float(np.mean(np.abs(rough.astype(np.float32) - target_bgr.astype(np.float32))))
    print(f"[faceswap] rough Δ={diff0:.2f}", flush=True)

    # facemesh-маска -> если пусто, возвращаем rough (чтобы своп был виден)
    base_mask, chin_y = self._facemesh_mask(rough)
    if base_mask.max() == 0:
        print("[faceswap] facemesh mask empty -> return rough", flush=True)
        return rough

    ext_mask = self._extended_mask(base_mask, chin_y)
    matched = self._reinhard_to_ref(rough, target_bgr, ext_mask)

    out = matched
    if self.cfg.use_hair_from_src:
        src_for_blend = src_face_bgr
        if src_for_blend.shape[:2] != out.shape[:2]:
            src_for_blend = cv2.resize(src_for_blend, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        h, w = out.shape[:2]
        upper = np.zeros_like(ext_mask)
        y0 = int(0.33*h) if chin_y is None else max(0, int(chin_y*0.35))
        upper[:y0, :] = 255
        hair_band = cv2.bitwise_and(ext_mask, upper)
        hair_band = cv2.GaussianBlur(hair_band, (0,0), 1.2)
        a = (hair_band.astype(np.float32)/255.0)[:,:,None] * float(self.cfg.alpha)
        out = (src_for_blend.astype(np.float32)*a + out.astype(np.float32)*(1-a)).astype(np.uint8)

    # вернуть очки/грани
    gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    glasses_mask = cv2.bitwise_and(edges, base_mask)
    glasses_mask = cv2.GaussianBlur(glasses_mask, (0,0), 1.2)
    glasses_a = (glasses_mask.astype(np.float32)/255.0)[:,:,None] * 0.85

    out = self._unsharp(out, sigma=0.8, amount=self.swap_sharp_gain)

    if self.poisson_full:
        ys, xs = np.where(ext_mask>0)
        if len(xs) > 0:
            center = (int(xs.mean()), int(ys.mean()))
            try:
                out = cv2.seamlessClone(out, target_bgr, ext_mask, center, cv2.MIXED_CLONE)
            except Exception as e:
                print(f"[faceswap][poisson_full] {e}", flush=True)

    elif self.cfg.poisson_face:
        band = cv2.morphologyEx(ext_mask, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        ys, xs = np.where(ext_mask>0)
        if len(xs) > 0 and band.max() > 0:
            center = (int(xs.mean()), int(ys.mean()))
            try:
                out = cv2.seamlessClone(out, target_bgr, band, center, cv2.MIXED_CLONE)
            except Exception as e:
                print(f"[faceswap][poisson_edge] {e}", flush=True)

    if glasses_a.max() > 0:
        out = (target_bgr.astype(np.float32)*glasses_a + out.astype(np.float32)*(1-glasses_a)).astype(np.uint8)

    diff1 = float(np.mean(np.abs(out.astype(np.float32) - target_bgr.astype(np.float32))))
    print(f"[faceswap] final Δ={diff1:.2f}", flush=True)

    if os.getenv("SAVE_DEBUG","0") == "1":
        try:
            cv2.imwrite("/tmp/debug_faceswap_rough.jpg", rough)
            cv2.imwrite("/tmp/debug_faceswap_mask_base.png", base_mask)
            cv2.imwrite("/tmp/debug_faceswap_mask_ext.png", ext_mask)
            cv2.imwrite("/tmp/debug_faceswap_out.jpg", out)
        except Exception: pass

    return out
