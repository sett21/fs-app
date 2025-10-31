import os

import cv2
import insightface
import numpy as np
import mediapipe as mp

from gen.utils.utils import _HairCfg, texture_reinject

# если _FACE_OVAL не импортируется из utils — раскомментируй и подставь свой список индексов
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109, 10,
]
_FACE_CHIN_IDX = 152  # подбородок (FaceMesh)
# FaceMesh: внешний контур губ (468‑точечная разметка)
_LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291]


def _ellipse_kernel(rx: int, ry: int) -> np.ndarray:
    rx = max(1, int(rx))
    ry = max(1, int(ry))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rx * 2 + 1, ry * 2 + 1))


class FaceSwapper:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face model not found: {model_path}")

        home = os.getenv("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))
        os.makedirs(os.path.join(home, "models"), exist_ok=True)

        self.home = home
        self.ctx = -1 if os.getenv("USE_CPU", "0") == "1" else 0
        self.det_thresh_primary = float(os.getenv("INSIGHTFACE_DET_THRESH", "0.5"))
        self.det_thresh_fallback = float(os.getenv("INSIGHTFACE_DET_THRESH_MIN", "0.25"))

        det_primary = int(os.getenv("INSIGHTFACE_DET_SIZE", "896"))
        det_sizes_env = os.getenv("INSIGHTFACE_DET_SIZES")
        det_sizes = []
        if det_sizes_env:
            for chunk in det_sizes_env.replace(";", ",").split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    val = int(chunk)
                except ValueError:
                    continue
                if val > 0:
                    det_sizes.append(val)
        else:
            det_sizes = [det_primary, 1024, 896, 768, 640]

        if det_primary > 0 and det_primary not in det_sizes:
            det_sizes.insert(0, det_primary)

        # гарантируем fallback-набор, самый маленький — 640
        if 640 not in det_sizes:
            det_sizes.append(640)

        seen = set()
        self.det_sizes = []
        for val in det_sizes:
            if val <= 0 or val in seen:
                continue
            seen.add(val)
            self.det_sizes.append(val)

        # Детекторы храним в кэше по размеру
        self._detectors: dict[int, insightface.app.FaceAnalysis] = {}

        providers = (["CPUExecutionProvider"] if self.ctx == -1 else ["CUDAExecutionProvider", "CPUExecutionProvider"])
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
                    min_detection_confidence=0.4,
                )
            except Exception:
                self.fm = None

        # Параметры смешивания/волос
        self.cfg = _HairCfg(
            use_hair_from_src=os.getenv("HAIR_FROM_SRC", "1") == "1",
            band_inner_px=int(os.getenv("HAIR_BAND_INNER_PX", "14")),
            band_outer_px=int(os.getenv("HAIR_BAND_OUTER_PX", "28")),
            alpha=float(os.getenv("HAIR_ALPHA", "0.85")),
            poisson_face=os.getenv("POISSON_FACE", "1") == "1",
        )

        # ENV-параметры для расширенной маски
        self.expand_ears_px = int(os.getenv("SWAP_EAR_EXPAND_PX", "18"))
        self.expand_hair_px = int(os.getenv("SWAP_HAIR_EXPAND_PX", "12"))
        self.chin_cut_px = int(os.getenv("SWAP_CHIN_CUT_PX", "6"))
        self.poisson_full = os.getenv("SWAP_POISSON_FULL", "1") == "1"
        self.swap_sharp_gain = float(os.getenv("SWAP_SHARP_GAIN", "0.25"))
        self.swap_select = os.getenv("SWAP_SELECT", "largest").strip().lower()
        self.swap_index = max(0, int(os.getenv("SWAP_INDEX", "0")))
        self.force_full_overlay = os.getenv("SWAP_FORCE_FULL", "0") == "1"
        # Тон‑матч смешение: 1.0 = только matched, 0.0 = только rough
        try:
            self.tonematch_gain = float(os.getenv("SWAP_TONEMATCH_GAIN", "1.0"))
        except Exception:
            self.tonematch_gain = 1.0
        # Смягчение области губ (убрать «двойные» границы/усы)
        self.mouth_soften = os.getenv("SWAP_MOUTH_SOFTEN", "1") == "1"
        try:
            self.mouth_band_px = int(os.getenv("SWAP_MOUTH_BAND_PX", "6"))
            self.mouth_blend   = float(os.getenv("SWAP_MOUTH_BLEND", "0.35"))
        except Exception:
            self.mouth_band_px, self.mouth_blend = 6, 0.35
        # Режим: сохранять уши таргета (меняем только ядро лица)
        self.keep_ears = os.getenv("SWAP_KEEP_EARS", "0") == "1"
        self.core_shrink_px = int(os.getenv("SWAP_CORE_SHRINK_PX", "4"))
        # Жёсткая замена ушей (после расширения маски)
        self.ear_hard = os.getenv("SWAP_EAR_HARD", "1") == "1"
        self.ear_hard_feather = int(os.getenv("SWAP_EAR_FEATHER", "4"))
        try:
            self.core_dilate = int(os.getenv("SWAP_CORE_DILATE", "11"))
        except Exception:
            self.core_dilate = 11
        # Дополнительный ear‑clone для устранения наложений
        self.ear_clone = os.getenv("SWAP_EAR_CLONE", "1") == "1"
        self.ear_clone_mode = os.getenv("SWAP_EAR_CLONE_MODE", "normal").strip().lower()
        self.ear_erode = max(0, int(os.getenv("SWAP_EAR_ERODE", "2")))
        self.ear_min_area = max(50, int(os.getenv("SWAP_EAR_MIN_AREA", "300")))
        # Dehalo по краю уха (мягкое притягивание к таргету)
        self.ear_dehalo = os.getenv("SWAP_EAR_DEHALO", "1") == "1"
        try:
            self.ear_dehalo_gain = float(os.getenv("SWAP_EAR_DEHALO_GAIN", "0.22"))
            self.ear_dehalo_sigma = float(os.getenv("SWAP_EAR_DEHALO_SIGMA", "1.6"))
        except Exception:
            self.ear_dehalo_gain, self.ear_dehalo_sigma = 0.22, 1.6
        # Очки/резкие края вокруг глаз: по умолчанию отключаем сильное подмешивание таргета
        try:
            self.glasses_alpha = float(os.getenv("SWAP_GLASSES_ALPHA", "0.0"))
        except Exception:
            self.glasses_alpha = 0.0
        # Исключать область рта из реинжекта высоких частот (убирает «двойные» усы/губы)
        self.detail_exclude_mouth = os.getenv("SWAP_DETAIL_EXCLUDE_MOUTH", "1") == "1"
        try:
            self.detail_mouth_scale = float(os.getenv("SWAP_DETAIL_MOUTH_SCALE", "0.2"))
        except Exception:
            self.detail_mouth_scale = 0.2
        # Fallback-маска, если FaceMesh дал смещение/пусто
        try:
            self.bbox_pad_pct = float(os.getenv("SWAP_BBOX_OVAL_PAD_PCT", "0.12"))
        except Exception:
            self.bbox_pad_pct = 0.12
        # Доп. ухо-настройки
        self.ear_expand_mult = float(os.getenv("SWAP_EAR_MULT", "1.6"))
        self.ear_extra_iters = max(1, int(os.getenv("SWAP_EAR_ITERS", "2")))
        self.force_ears = os.getenv("SWAP_FORCE_EARS", "1") == "1"

    # ---------- Утилиты ----------
    def _get_detector(self, det_size: int) -> insightface.app.FaceAnalysis:
        det = self._detectors.get(det_size)
        if det is None:
            det = insightface.app.FaceAnalysis(name="buffalo_l", root=self.home)
            det.prepare(ctx_id=self.ctx, det_size=(det_size, det_size))
            try:
                det.det_thresh = self.det_thresh_primary
            except Exception:
                pass
            self._detectors[det_size] = det
        return det

    def _rescale_faces(self, faces, scale: float):
        if scale == 1.0 or not faces:
            return faces
        out = []
        for f in faces:
            g = type(f)()
            g.bbox = (f.bbox / scale).astype(f.bbox.dtype)
            g.kps = (f.kps / scale).astype(f.kps.dtype) if getattr(f, "kps", None) is not None else None
            g.det_score = getattr(f, "det_score", None)
            out.append(g)
        return out

    def _detect_faces(self, img: np.ndarray):
        for det_size in self.det_sizes:
            detector = self._get_detector(det_size)
            faces = self._run_detector(detector, img, det_size)
            if faces:
                suffix = " (fallback)" if det_size != self.det_sizes[0] else ""
                print(f"[faceswap] detector det_size={det_size} faces={len(faces)}{suffix}", flush=True)
                return faces
        return []

    def _run_detector(self, detector: insightface.app.FaceAnalysis, img: np.ndarray, det_size: int):
        det_thresh_bak = getattr(detector, "det_thresh", None)

        faces = detector.get(img)
        if faces:
            return faces

        # fallback: слегка опускаем порог, если он выше заданного минимума
        min_thresh = self.det_thresh_fallback
        if min_thresh is not None and det_thresh_bak is not None and min_thresh < det_thresh_bak:
            try:
                detector.det_thresh = min_thresh
            except Exception:
                pass
            faces = detector.get(img)
            if faces:
                if det_thresh_bak is not None:
                    try:
                        detector.det_thresh = det_thresh_bak
                    except Exception:
                        pass
                return faces

        h, w = img.shape[:2]
        scale = 1.6 if max(h, w) < 1400 else 1.2
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        faces = detector.get(big)
        if faces:
            if det_thresh_bak is not None:
                try:
                    detector.det_thresh = det_thresh_bak
                except Exception:
                    pass
            return self._rescale_faces(faces, scale)

        yuv = cv2.cvtColor(big, cv2.COLOR_BGR2YUV)
        y = yuv[..., 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        yuv[..., 0] = y
        big2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        sharp = cv2.GaussianBlur(big2, (0, 0), 1.2)
        big2 = np.clip(big2.astype(np.float32) + 0.6 * (big2.astype(np.float32) - sharp.astype(np.float32)), 0, 255).astype(np.uint8)
        faces = detector.get(big2)

        if det_thresh_bak is not None:
            try:
                detector.det_thresh = det_thresh_bak
            except Exception:
                pass

        if faces:
            return self._rescale_faces(faces, scale)
        if det_thresh_bak is not None:
            try:
                detector.det_thresh = det_thresh_bak
            except Exception:
                pass
        return []

    def _pick_target_face(self, faces, img_shape=None, prefer_xy=None):
        if not faces:
            return None

        mode = self.swap_select
        if mode == "index":
            idx = min(self.swap_index, len(faces) - 1)
            return faces[idx]

        if mode == "center" and prefer_xy is None and img_shape is not None:
            h, w = img_shape
            prefer_xy = (w / 2.0, h / 2.0)

        if prefer_xy is not None:
            qx, qy = prefer_xy

            def _dist(f):
                x0, y0, x1, y1 = f.bbox.astype(int)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                return (cx - qx) ** 2 + (cy - qy) ** 2

            return sorted(faces, key=_dist)[0]

        def score(f):
            x1, y1, x2, y2 = f.bbox.astype(int)
            area = max(1, (x2 - x1) * (y2 - y1))
            kps = f.kps
            if kps is None or kps.shape[0] < 2:
                fr = 1.0
            else:
                d = np.linalg.norm(kps[0] - kps[1]) + 1e-6
                v = abs((kps[0][1] - kps[1][1])) / d
                fr = 1.0 - min(1.0, v)
            return area * (1.0 + 0.3 * fr)

        if mode == "largest":
            return sorted(faces, key=score, reverse=True)[0]

        return faces[0]

    def _facemesh_mask(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        chin_y = None
        if self.fm is None:
            return mask, chin_y
        res = self.fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res or not res.multi_face_landmarks:
            return mask, chin_y
        lm = res.multi_face_landmarks[0]
        chin_y = int(lm.landmark[_FACE_CHIN_IDX].y * h)
        pts = np.array([[int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)] for i in _FACE_OVAL], np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        return mask, chin_y

    def _mouth_band_mask(self, bgr: np.ndarray, px: int):
        h, w = bgr.shape[:2]
        m = np.zeros((h, w), np.uint8)
        if self.fm is None:
            return m
        try:
            res = self.fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            return m
        if not res or not res.multi_face_landmarks:
            return m
        lm = res.multi_face_landmarks[0]
        pts = np.array([[int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)] for i in _LIPS_OUTER], np.int32)
        if pts.shape[0] >= 3:
            cv2.fillConvexPoly(m, pts, 255)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
            band = cv2.dilate(m, k, 1)
            return band
        return m

    def _bbox_oval_mask(self, shape_hw, bbox):
        h, w = shape_hw
        x0, y0, x1, y1 = [int(v) for v in bbox]
        bw = max(8, x1 - x0)
        bh = max(8, y1 - y0)
        pad_x = int(self.bbox_pad_pct * bw)
        pad_y = int(self.bbox_pad_pct * bh)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        ax = max(6, (bw // 2) + pad_x)
        ay = max(6, (bh // 2) + pad_y)
        m = np.zeros((h, w), np.uint8)
        try:
            cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        except Exception:
            cv2.rectangle(m, (max(0, x0 - pad_x), max(0, y0 - pad_y)), (min(w - 1, x1 + pad_x), min(h - 1, y1 + pad_y)), 255, -1)
        return m

    def _extended_mask(self, base_mask: np.ndarray, chin_y: int | None, bbox=None):
        if base_mask.max() == 0:
            return base_mask
        # Если нужно оставить уши оригинала — не расширяем маску, а наоборот слегка сужаем ядро
        if self.keep_ears:
            if self.core_shrink_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.core_shrink_px*2+1, self.core_shrink_px*2+1))
                core = cv2.erode(base_mask, k, 1)
            else:
                core = base_mask
            return core
        # Более агрессивное расширение по X (уши)
        rx = max(1, int(self.expand_ears_px * self.ear_expand_mult))
        ry = max(2, rx // 3)
        ear_k = _ellipse_kernel(rx, ry)
        ext = cv2.dilate(base_mask, ear_k, self.ear_extra_iters)
        # Доп. прямоугольное расширение в стороны — стабилизирует покрытие уха
        wide_k = cv2.getStructuringElement(cv2.MORPH_RECT, (rx * 2 + 1, max(3, (rx // 2) * 2 + 1)))
        ext = cv2.dilate(ext, wide_k, 1)
        # Волосы — вверх по Y
        hair_k = _ellipse_kernel(max(2, self.expand_hair_px // 3), self.expand_hair_px)
        ext = cv2.dilate(ext, hair_k, 1)
        # Добавим прямоугольные «карманы» в области ушей по bbox лица
        try:
            if bbox is not None:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                h, w = base_mask.shape[:2]
                x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1))
                y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1))
                # Вертикальный диапазон ушей: от верхней границы маски до подбородка
                ys, xs = np.where(base_mask > 0)
                if len(ys) > 0:
                    top = int(np.min(ys))
                else:
                    top = y0
                ear_y0 = max(0, top)
                ear_y1 = min(h - 1, chin_y if chin_y is not None else y1)
                # Ширина «кармана» берётся от env или из rx
                rect_w = int(os.getenv("SWAP_EAR_RECT_W", "16"))
                # Левый карман
                lx0 = max(0, x0 - rect_w)
                lx1 = max(0, min(w - 1, x0 + rect_w))
                ext[ear_y0:ear_y1, lx0:lx1] = 255
                # Правый карман
                rx0 = max(0, min(w - 1, x1 - rect_w))
                rx1 = min(w - 1, x1 + rect_w)
                ext[ear_y0:ear_y1, rx0:rx1] = 255
        except Exception:
            pass
        if chin_y is not None and self.chin_cut_px > 0:
            y_cut = min(ext.shape[0] - 1, max(0, chin_y + self.chin_cut_px))
            ext[y_cut:, :] = 0
        return ext

    def _reinhard_to_ref(self, src_img, ref_img, msk):
        m = msk > 0
        if not np.any(m):
            return src_img
        A = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        B = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in range(3):
            mu_a, sd_a = A[..., c][m].mean(), A[..., c][m].std() + 1e-6
            mu_b, sd_b = B[..., c][m].mean(), B[..., c][m].std() + 1e-6
            A[..., c][m] = (A[..., c][m] - mu_a) * (sd_b / sd_a) + mu_b
        return cv2.cvtColor(np.clip(A, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _unsharp(self, img, sigma=0.8, amount=0.25):
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return np.clip(img.astype(np.float32) + amount * (img.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)

    # ---------- Основной метод ----------
    def swap(self, target_bgr: np.ndarray, src_face_bgr: np.ndarray, quad_center=None, avoid_poly=None) -> np.ndarray:
        if target_bgr is None or src_face_bgr is None:
            print("[faceswap] empty inputs", flush=True)
            return target_bgr

        tar_faces = self._detect_faces(target_bgr)
        src_faces = self._detect_faces(src_face_bgr)
        print(f"[faceswap] detected target={len(tar_faces)} src={len(src_faces)}", flush=True)

        # Если задан полигон области, которую нужно избегать (например, область открытки),
        # исключаем лица, центры которых попадают внутрь него.
        if avoid_poly is not None and len(tar_faces) > 0:
            try:
                poly = np.asarray(avoid_poly, dtype=np.float32)
                if poly.shape == (4, 2):
                    kept = []
                    for f in tar_faces:
                        x0, y0, x1, y1 = f.bbox.astype(int)
                        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                        inside = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
                        if not inside:
                            kept.append(f)
                    if kept:
                        print(f"[faceswap] avoid_poly applied: {len(tar_faces)} -> {len(kept)}", flush=True)
                        tar_faces = kept
                    else:
                        print("[faceswap] avoid_poly excluded all faces; keeping original list", flush=True)
                else:
                    print("[faceswap] avoid_poly ignored: wrong shape", flush=True)
            except Exception as e:
                print(f"[faceswap] avoid_poly error: {e}", flush=True)

        if not src_faces:
            if os.getenv("SAVE_DEBUG", "0") == "1":
                try:
                    cv2.imwrite("/tmp/debug_src_no_face.jpg", src_face_bgr)
                except Exception:
                    pass
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

        tx1, ty1, tx2, ty2 = tar.bbox.astype(int)
        sx1, sy1, sx2, sy2 = src.bbox.astype(int)
        print(f"[faceswap] target bbox=({tx1},{ty1},{tx2},{ty2}) src bbox=({sx1},{sy1},{sx2},{sy2})", flush=True)

        rough = self.swapper.get(target_bgr.copy(), tar, src, paste_back=True)
        if rough is None:
            print("[faceswap] swapper returned None", flush=True)
            return target_bgr

        diff0 = float(np.mean(np.abs(rough.astype(np.float32) - target_bgr.astype(np.float32))))
        print(f"[faceswap] rough Δ={diff0:.2f}", flush=True)

        if self.force_full_overlay:
            print("[faceswap] SWAP_FORCE_FULL=1 -> используем результат swapper без маски", flush=True)
            if os.getenv("SAVE_DEBUG", "0") == "1":
                try:
                    cv2.imwrite("/tmp/debug_faceswap_force_full.jpg", rough)
                except Exception:
                    pass
            return rough

        base_mask, chin_y = self._facemesh_mask(rough)
        # Фолбэк: если маска пустая или не попадает в bbox — берём овал по bbox
        if base_mask.max() == 0:
            base_mask = self._bbox_oval_mask(rough.shape[:2], tar.bbox)
            print("[faceswap] facemesh empty -> bbox oval", flush=True)
        else:
            # Проверим покрытие bbox
            x0, y0, x1, y1 = tar.bbox.astype(int)
            x0 = max(0, x0); y0 = max(0, y0); x1 = min(base_mask.shape[1]-1, x1); y1 = min(base_mask.shape[0]-1, y1)
            bbox_area = max(1, (x1 - x0) * (y1 - y0))
            cover = int(base_mask[y0:y1, x0:x1].sum() // 255)
            ratio = cover / float(bbox_area)
            if ratio < 0.35:
                base_mask = self._bbox_oval_mask(rough.shape[:2], tar.bbox)
                print(f"[faceswap] facemesh->bbox fallback ratio={ratio:.2f}", flush=True)

        ext_mask = self._extended_mask(base_mask, chin_y, bbox=tar.bbox)
        matched = self._reinhard_to_ref(rough, target_bgr, ext_mask)

        # Смешиваем rough и matched для большей похожести на rough
        if 0.0 <= self.tonematch_gain < 1.0:
            m01 = (ext_mask.astype(np.float32) / 255.0)[:, :, None]
            mix = self.tonematch_gain
            blend = matched.astype(np.float32) * mix + rough.astype(np.float32) * (1.0 - mix)
            out = (blend * m01 + target_bgr.astype(np.float32) * (1.0 - m01)).astype(np.uint8)
        else:
            out = matched
        # Жёстко заменить уши: (отключаем, если keep_ears)
        if self.ear_hard and not self.keep_ears:
            try:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.core_dilate, self.core_dilate))
                core = cv2.dilate(base_mask, k, 1)
                ear_only = cv2.subtract(ext_mask, core)
                if ear_only.max() > 0:
                    sigma = max(0.1, float(self.ear_hard_feather))
                    a = cv2.GaussianBlur(ear_only, (0, 0), sigma)
                    a = (a.astype(np.float32) / 255.0)[:, :, None]
                    out = (matched.astype(np.float32) * a + out.astype(np.float32) * (1 - a)).astype(np.uint8)
                    print("[faceswap] ear_hard blend applied", flush=True)
            except Exception as e:
                print(f"[faceswap][ear_hard] {e}", flush=True)
        if self.cfg.use_hair_from_src:
            src_for_blend = src_face_bgr
            if src_for_blend.shape[:2] != out.shape[:2]:
                src_for_blend = cv2.resize(src_for_blend, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
            h, w = out.shape[:2]
            upper = np.zeros_like(ext_mask)
            y0 = int(0.33 * h) if chin_y is None else max(0, int(chin_y * 0.35))
            upper[:y0, :] = 255
            hair_band = cv2.bitwise_and(ext_mask, upper)
            hair_band = cv2.GaussianBlur(hair_band, (0, 0), 1.2)
            a = (hair_band.astype(np.float32) / 255.0)[:, :, None] * float(self.cfg.alpha)
            out = (src_for_blend.astype(np.float32) * a + out.astype(np.float32) * (1 - a)).astype(np.uint8)

        gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
        glasses_mask = cv2.bitwise_and(edges, base_mask)
        glasses_mask = cv2.GaussianBlur(glasses_mask, (0, 0), 1.2)
        # Управляемая сила возврата таргета по «очкам». 0 — отключено
        ga = max(0.0, float(self.glasses_alpha))
        glasses_a = (glasses_mask.astype(np.float32) / 255.0)[:, :, None] * ga

        out = self._unsharp(out, sigma=0.8, amount=self.swap_sharp_gain)
        # Подмешиваем высокочастотные детали таргета, чтобы убрать «пластик» и швы
        try:
            det_gain  = float(os.getenv("SWAP_DETAIL_GAIN", "0.22"))
            det_sigma = float(os.getenv("SWAP_DETAIL_SIGMA", "0.9"))
            if det_gain > 0:
                m01 = (ext_mask.astype(np.float32) / 255.0)
                # В зоне рта ослабляем реинжект, чтобы не появлялись «усы/двойные губы»
                if self.detail_exclude_mouth and self.mouth_band_px > 0 and self.fm is not None:
                    try:
                        band = self._mouth_band_mask(rough, max(2, self.mouth_band_px))
                        if band is not None and band.max() > 0:
                            a = cv2.GaussianBlur(band, (0,0), 1.2).astype(np.float32)/255.0
                            m01 = np.clip(m01 * (1.0 - a * float(self.detail_mouth_scale)), 0.0, 1.0)
                    except Exception:
                        pass
                out = texture_reinject(out, target_bgr, m01, sigma=det_sigma, gain=det_gain)
        except Exception:
            pass

        # Смягчение зоны губ — смешиваем с таргетом в узкой полосе
        if self.mouth_soften and self.mouth_blend > 0:
            band = self._mouth_band_mask(out, max(2, self.mouth_band_px))
            if band.max() > 0:
                a = cv2.GaussianBlur(band, (0,0), 1.2).astype(np.float32)/255.0
                a = (a * float(self.mouth_blend))[:, :, None]
                out = (target_bgr.astype(np.float32) * a + out.astype(np.float32) * (1.0 - a)).astype(np.uint8)

        use_poisson_full = (self.poisson_full and not self.keep_ears)
        if use_poisson_full:
            ys, xs = np.where(ext_mask > 0)
            if len(xs) > 0:
                center = (int(xs.mean()), int(ys.mean()))
                try:
                    out = cv2.seamlessClone(out, target_bgr, ext_mask, center, cv2.MIXED_CLONE)
                except Exception as e:
                    print(f"[faceswap][poisson_full] {e}", flush=True)
        elif self.cfg.poisson_face:
            band = cv2.morphologyEx(ext_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            ys, xs = np.where(ext_mask > 0)
            if len(xs) > 0 and band.max() > 0:
                center = (int(xs.mean()), int(ys.mean()))
                try:
                    out = cv2.seamlessClone(out, target_bgr, band, center, cv2.MIXED_CLONE)
                except Exception as e:
                    print(f"[faceswap][poisson_edge] {e}", flush=True)
        elif self.force_ears and not self.keep_ears:
            # Принудительная перерисовка ушей полноразмерным Poisson, если base-пути отключены
            ys, xs = np.where(ext_mask > 0)
            if len(xs) > 0:
                center = (int(xs.mean()), int(ys.mean()))
                try:
                    out = cv2.seamlessClone(out, target_bgr, ext_mask, center, cv2.MIXED_CLONE)
                    print("[faceswap] force_ears Poisson applied", flush=True)
                except Exception as e:
                    print(f"[faceswap][force_ears] {e}", flush=True)

        # Локальный clone только для каждой «ушной» компоненты, чтобы убрать двойные уши
        if self.ear_clone and not self.keep_ears:
            try:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.core_dilate, self.core_dilate))
                core = cv2.dilate(base_mask, k, 1)
                ear_only = cv2.subtract(ext_mask, core)
                if self.ear_erode > 0:
                    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ear_erode*2+1, self.ear_erode*2+1))
                    ear_only = cv2.erode(ear_only, ek, 1)
                cnts, _ = cv2.findContours(ear_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mode = cv2.NORMAL_CLONE if self.ear_clone_mode != "mixed" else cv2.MIXED_CLONE
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < self.ear_min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    mask = np.zeros_like(ear_only)
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    M = cv2.moments(c)
                    if M['m00'] == 0:
                        continue
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    try:
                        out = cv2.seamlessClone(out, target_bgr, mask, (cx, cy), mode)
                    except Exception:
                        pass
                print("[faceswap] ear_clone components applied", flush=True)
            except Exception as e:
                print(f"[faceswap][ear_clone] {e}", flush=True)

        # Гарантируем, что центральное ядро остаётся из src (если Poisson смягчил слишком сильно)
        try:
            if os.getenv("SWAP_ENFORCE_CORE", "1") == "1":
                core_k = max(3, int(os.getenv("SWAP_CORE_DILATE", str(self.core_dilate))))
                feather = float(os.getenv("SWAP_CORE_FEATHER", "2.0"))
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_k, core_k))
                core = cv2.erode(base_mask, k, 1)
                a = cv2.GaussianBlur(core, (0, 0), max(0.1, feather)).astype(np.float32) / 255.0
                out = (matched.astype(np.float32) * a[:, :, None] + out.astype(np.float32) * (1 - a[:, :, None])).astype(np.uint8)
                print("[faceswap] enforce core blend", flush=True)
        except Exception as e:
            print(f"[faceswap][enforce_core] {e}", flush=True)

        if glasses_a.max() > 0:
            out = (target_bgr.astype(np.float32) * glasses_a + out.astype(np.float32) * (1 - glasses_a)).astype(np.uint8)

        # Мягкий dehalo по краю уха: притягиваем край к таргету
        if self.ear_dehalo and not self.keep_ears:
            try:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.core_dilate, self.core_dilate))
                core = cv2.dilate(base_mask, k, 1)
                ear_only = cv2.subtract(ext_mask, core)
                band = cv2.morphologyEx(ear_only, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
                if band.max() > 0 and self.ear_dehalo_gain > 0:
                    a = cv2.GaussianBlur(band, (0,0), max(0.1, self.ear_dehalo_sigma)).astype(np.float32) / 255.0
                    a = (a * float(self.ear_dehalo_gain))[:, :, None]
                    out = (target_bgr.astype(np.float32) * a + out.astype(np.float32) * (1.0 - a)).astype(np.uint8)
            except Exception:
                pass

        diff1 = float(np.mean(np.abs(out.astype(np.float32) - target_bgr.astype(np.float32))))
        print(f"[faceswap] final Δ={diff1:.2f}", flush=True)

        if os.getenv("SAVE_DEBUG", "0") == "1":
            try:
                cv2.imwrite("/tmp/debug_faceswap_rough.jpg", rough)
                cv2.imwrite("/tmp/debug_faceswap_mask_base.png", base_mask)
                cv2.imwrite("/tmp/debug_faceswap_mask_ext.png", ext_mask)
                cv2.imwrite("/tmp/debug_faceswap_out.jpg", out)
            except Exception:
                pass

        return out
