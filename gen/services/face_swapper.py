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
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face model not found: {model_path}")

        home = os.getenv("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))
        os.makedirs(os.path.join(home, "models"), exist_ok=True)

        # Детектор лиц
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", root=home)
        ctx = -1 if os.getenv("USE_CPU","0") == "1" else 0
        det = int(os.getenv("INSIGHTFACE_DET_SIZE", "896"))
        self.app.prepare(ctx_id=ctx, det_size=(det, det))
        try:
            self.app.det_thresh = float(os.getenv("INSIGHTFACE_DET_THRESH", "0.5"))
        except Exception:
            pass

        providers = (['CPUExecutionProvider'] if ctx == -1
                     else ['CUDAExecutionProvider','CPUExecutionProvider'])
        self.swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        print("[insightface] ORT providers:", providers, flush=True)

        # FaceMesh
        if os.getenv("DISABLE_FACEMESH", "0") == "1":
            self.fm = None
        else:
            try:
                self.fm = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=2,
                    min_detection_confidence=0.4
                )
            except Exception:
                self.fm = None

        # Параметры смешивания/волос
        self.cfg = _HairCfg(
            use_hair_from_src = os.getenv("HAIR_FROM_SRC","1") == "1",
            band_inner_px     = int(os.getenv("HAIR_BAND_INNER_PX","14")),
            band_outer_px     = int(os.getenv("HAIR_BAND_OUTER_PX","28")),
            alpha             = float(os.getenv("HAIR_ALPHA","0.85")),
            poisson_face      = os.getenv("POISSON_FACE","1") == "1",
        )

        # Новые ENV-параметры для расширенной маски
        self.expand_ears_px  = int(os.getenv("SWAP_EAR_EXPAND_PX", "18"))
        self.expand_hair_px  = int(os.getenv("SWAP_HAIR_EXPAND_PX", "12"))
        self.chin_cut_px     = int(os.getenv("SWAP_CHIN_CUT_PX", "6"))
        self.poisson_full    = os.getenv("SWAP_POISSON_FULL", "1") == "1"
        self.swap_sharp_gain = float(os.getenv("SWAP_SHARP_GAIN", "0.25"))

    # ---------- Внутрянка ----------
    def _pick_target_face(self, faces, prefer_xy=None):
        """Выбираем лицо: либо ближайшее к prefer_xy, либо крупное/фронтальное."""
        if not faces:
            return None
        if prefer_xy is not None:
            qx, qy = prefer_xy
            def _dist(f):
                x0,y0,x1,y1 = f.bbox.astype(int)
                cx, cy = (x0+x1)/2, (y0+y1)/2
                return (cx-qx)**2 + (cy-qy)**2
            return sorted(faces, key=_dist)[0]
        # fallback: площадь * фронтальность
        def score(f):
            x1,y1,x2,y2 = f.bbox.astype(int)
            area = max(1,(x2-x1)*(y2-y1))
            kps = f.kps
            if kps is None or kps.shape[0] < 2:
                fr = 1.0
            else:
                d = np.linalg.norm(kps[0]-kps[1]) + 1e-6
                v = abs((kps[0][1]-kps[1][1]))/d
                fr = 1.0 - min(1.0, v)
            return area*(1.0 + 0.3*fr)
        return sorted(faces, key=score, reverse=True)[0]

    def _facemesh_mask(self, bgr: np.ndarray):
        """Возвращает (base_mask, chin_y). base_mask — овал из FaceMesh."""
        h, w = bgr.shape[:2]
        mask = np.zeros((h,w), np.uint8)
        chin_y = None
        if self.fm is None:
            return mask, chin_y
        res = self.fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res or not res.multi_face_landmarks:
            return mask, chin_y
        lm = res.multi_face_landmarks[0]
        chin_y = int(lm.landmark[_FACE_CHIN_IDX].y * h)
        pts = np.array([[int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)] for i in _FACE_OVAL], np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        return mask, chin_y

    def _extended_mask(self, base_mask: np.ndarray, chin_y: int | None):
        """Анизотропно расширяем: по X — уши, по Y вверх — волосы. Низ обрезаем по подбородку."""
        if base_mask.max()==0:
            return base_mask
        ear_k  = _ellipse_kernel(self.expand_ears_px, max(2, self.expand_ears_px//3))
        hair_k = _ellipse_kernel(max(2, self.expand_hair_px//3), self.expand_hair_px)
        ext = cv2.dilate(base_mask, ear_k, 1)
        ext = cv2.dilate(ext, hair_k, 1)
        if chin_y is not None and self.chin_cut_px > 0:
            y_cut = min(ext.shape[0]-1, max(0, chin_y + self.chin_cut_px))
            ext[y_cut:, :] = 0
        return ext

    def _reinhard_to_ref(self, src_img, ref_img, msk):
        m = msk > 0
        if not np.any(m): return src_img
        A = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        B = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in range(3):
            mu_a, sd_a = A[...,c][m].mean(), A[...,c][m].std()+1e-6
            mu_b, sd_b = B[...,c][m].mean(), B[...,c][m].std()+1e-6
            A[...,c][m] = (A[...,c][m]-mu_a)*(sd_b/sd_a) + mu_b
        return cv2.cvtColor(np.clip(A,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _unsharp(self, img, sigma=0.8, amount=0.25):
        blur = cv2.GaussianBlur(img, (0,0), sigma)
        return np.clip(img.astype(np.float32) + amount*(img.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)

    # ---------- Основной метод ----------
    def swap(self, target_bgr: np.ndarray, src_face_bgr: np.ndarray, quad_center=None) -> np.ndarray:
        tar_faces = self.app.get(target_bgr)
        src_faces = self.app.get(src_face_bgr)
        if not tar_faces or not src_faces:
            return target_bgr

        tar = self._pick_target_face(tar_faces, prefer_xy=quad_center)
        src = self._pick_target_face(src_faces, prefer_xy=None)
        if tar is None or src is None:
            return target_bgr

        # базовый своп
        rough = self.swapper.get(target_bgr.copy(), tar, src, paste_back=True)

        # базовая маска + расширенная маска
        base_mask, chin_y = self._facemesh_mask(rough)
        if base_mask.max()==0:
            return rough
        ext_mask = self._extended_mask(base_mask, chin_y)

        # цветовая подгонка в расширенной маске
        matched = self._reinhard_to_ref(rough, target_bgr, ext_mask)

        # (опция) усилить волосы источника узкой полосой над лбом
        out = matched
        if self.cfg.use_hair_from_src:
            h, w = out.shape[:2]
            upper = np.zeros_like(ext_mask)
            y0 = int(0.33*h) if chin_y is None else max(0, int(chin_y*0.35))
            upper[:y0, :] = 255
            hair_band = cv2.bitwise_and(ext_mask, upper)
            hair_band = cv2.GaussianBlur(hair_band, (0,0), 1.2)
            a = (hair_band.astype(np.float32)/255.0)[:,:,None] * float(self.cfg.alpha)
            out = (src_face_bgr.astype(np.float32)*a + out.astype(np.float32)*(1-a)).astype(np.uint8)

        # вернуть очки/резкие края от таргета (если есть)
        gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
        glasses_mask = cv2.bitwise_and(edges, base_mask)
        glasses_mask = cv2.GaussianBlur(glasses_mask, (0,0), 1.2)
        glasses_a = (glasses_mask.astype(np.float32)/255.0)[:,:,None] * 0.85

        # лёгкая резкость по всему лицу после цветовой подгонки
        out = self._unsharp(out, sigma=0.8, amount=self.swap_sharp_gain)

        # полноразмерный Poisson по расширенной маске — переносит уши/волосы
        if self.poisson_full:
            ys, xs = np.where(ext_mask>0)
            if len(xs) > 0:
                center = (int(xs.mean()), int(ys.mean()))
                try:
                    out = cv2.seamlessClone(out, target_bgr, ext_mask, center, cv2.MIXED_CLONE)
                except Exception:
                    pass
        elif self.cfg.poisson_face:
            # fallback: узкий по границе, если полный отключён
            band = cv2.morphologyEx(ext_mask, cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
            ys, xs = np.where(ext_mask>0)
            if len(xs) > 0 and band.max() > 0:
                center = (int(xs.mean()), int(ys.mean()))
                try:
                    out = cv2.seamlessClone(out, target_bgr, band, center, cv2.MIXED_CLONE)
                except Exception:
                    pass

        # вернуть очки/грани из таргета
        if glasses_a.max() > 0:
            out = (target_bgr.astype(np.float32)*glasses_a + out.astype(np.float32)*(1-glasses_a)).astype(np.uint8)

        return out
