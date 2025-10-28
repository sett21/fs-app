import os

from gen.services.face_swapper import FaceSwapper
from gen.services.inpainter import Inpainter
from gen.utils.utils import add_fine_grain, apply_page_stack, bbox_from_mask, bgr_from_pil, build_contact_masks_tight, detect_green_quad, green_defringe, hand_mask_bgr, laplacian_blend, lin_to_srgb, pil_from_bgr, refine_alpha_with_edges, resize_limit, ring_grad, srgb_to_lin, texture_reinject, warp_into_quad
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import cv2, numpy as np, torch, time
from typing import Optional
from dataclasses import dataclass
from PIL import Image

torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "2"))))
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))
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

        # --- края / перья ---
        self.ring_in_pct  = float(os.getenv("RING_IN_PCT",  "0.004"))
        self.ring_out_pct = float(os.getenv("RING_OUT_PCT", "0.012"))
        self.seamless_edge = os.getenv("SEAMLESS_EDGE", "1") == "1"

        # --- эффект страницы ---
        self.page_effect   = os.getenv("PAGE_EFFECT","1") == "1"
        self.page_side     = os.getenv("PAGE_SIDE","right")
        self.page_inner_px = int(os.getenv("PAGE_INNER_PX","16"))
        self.page_shade    = float(os.getenv("PAGE_SHADE","0.06"))
        self.page_stripes  = os.getenv("PAGE_STRIPES","1") == "1"

        # --- faceswap ---
        self.disable_swap = os.getenv("DISABLE_FACE_SWAP","0") == "1"
        swap_mode_env = os.getenv("SWAP_MODE","").lower().strip()
        if not swap_mode_env:
            swap_mode_env = "before" if os.getenv("SWAP_BEFORE_COMPOSE","1") == "1" else "after"
        self.swap_mode = swap_mode_env

        # --- инициализация подсистем ---
        self.inpainter   = Inpainter(self.model_repo, self.dtype)
        self.swapper     = FaceSwapper(os.getenv("FACE_MODEL_PATH", "/models/inswapper_128.onnx"))

    def run(self, selfie_bgr: np.ndarray, postcard_bgr: np.ndarray, face_bgr: np.ndarray,
            prompt: Optional[str]=None) -> np.ndarray:
        t0 = time.time()
        if selfie_bgr is None or postcard_bgr is None:
            raise ValueError("Empty input: selfie or postcard is None")

        selfie_bgr, _ = resize_limit(selfie_bgr, MAX_SIDE)
        face_bgr,   _ = resize_limit(face_bgr,   MAX_SIDE)
        if os.getenv("KEEP_POSTCARD_RES", "1") != "1":
            postcard_bgr, _ = resize_limit(postcard_bgr, MAX_SIDE)

        H, W = selfie_bgr.shape[:2]

        # ===== 1) Прямоугольник =====
        quad = detect_green_quad(selfie_bgr, "#00ff84")

        if quad is None or quad.shape != (4,2):
            raise RuntimeError("Green rectangle (#00ff84) not found or invalid")

        quad_center = (float(quad[:,0].mean()), float(quad[:,1].mean()))
        swap_kwargs = {"quad_center": quad_center}

        # ===== 2) Faceswap (до) =====
        if not self.disable_swap and self.swap_mode == "before":
            print("[faceswap] mode=before", flush=True)
            try:
                selfie_bgr = self.swapper.swap(selfie_bgr, face_bgr, **swap_kwargs)
            except Exception as e:
                print(f"[faceswap][before] failed: {e}", flush=True)

        # ===== 3) «Книжный» край =====
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

        # ===== 4) Перспективная вклейка =====
        overlay, mask_poly = warp_into_quad(postcard_bgr, quad, (H, W))

        # ===== 5) Позади руки =====
        hand   = hand_mask_bgr(selfie_bgr)            # 255 = рука
        nohand = (hand == 0).astype(np.uint8)
        place = (nohand & (mask_poly>0).astype(np.uint8)).astype(np.float32)
        alpha_hard = place[:, :, None]

        # линейный свет
        base_lin = srgb_to_lin(selfie_bgr)
        ov_lin   = srgb_to_lin(overlay)
        init     = lin_to_srgb(ov_lin*alpha_hard + base_lin*(1-alpha_hard))

        # ===== 6) Узкий laplacian-blend =====
        rin  = max(1, int(self.ring_in_pct  * min(H, W)))
        rout = max(3, int(self.ring_out_pct * min(H, W)))
        edge_grad  = ring_grad(mask_poly, rin, rout)
        alpha_edge = (edge_grad[:, :, None] * (nohand[:, :, None]/255.0)).astype(np.float32)
        alpha_edge = np.clip(alpha_edge * 0.8, 0, 1)
        init = laplacian_blend(overlay, init, alpha_edge, levels=3)

        # ===== 7) Дефриндж зелени =====
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        edge_band  = cv2.morphologyEx(mask_poly, cv2.MORPH_GRADIENT, k3)
        under_hand = cv2.bitwise_and(hand, mask_poly)
        to_despill = cv2.bitwise_or(edge_band, under_hand)
        init = green_defringe(init, overlay, to_despill)

        # ===== 7.4) Контактный шов пальцев =====
        contact_line, edge_narrow = build_contact_masks_tight(hand, mask_poly)
        contact_alpha = (contact_line.astype(np.float32)/255.0)
        contact_alpha = refine_alpha_with_edges(selfie_bgr, contact_alpha, iters=2)
        shade_k = float(os.getenv("RIM_SHADE", "0.06"))
        mul = 1.0 - shade_k * contact_alpha[:, :, None]
        init = np.clip(init.astype(np.float32) * mul, 0, 255).astype(np.uint8)

        # лёгкая тень вдоль кромки для реализма
        edge_dist = cv2.distanceTransform((mask_poly>0).astype(np.uint8), cv2.DIST_L2, 3)
        if edge_dist.max() > 0:
            edge_dist = edge_dist / edge_dist.max()
        rim = np.clip(1.0 - edge_dist, 0, 1)
        rim = cv2.GaussianBlur(rim, (0,0), 1.0)
        rim = rim * (nohand.astype(np.float32))
        mul = 1.0 - shade_k * rim[:, :, None]
        init = np.clip(init.astype(np.float32) * mul, 0, 255).astype(np.uint8)

        mask_edge = np.clip(edge_narrow.astype(np.uint8) | (contact_alpha*255).astype(np.uint8), 0, 255)
        _, mask_edge = cv2.threshold(mask_edge, 1, 255, cv2.THRESH_BINARY)

        if self.seamless_edge:
            ys, xs = np.where(mask_poly > 0)
            if len(xs) > 0:
                center = (int(xs.mean()), int(ys.mean()))
                try:
                    init = cv2.seamlessClone(overlay, init, edge_band, center, cv2.MIXED_CLONE)
                except Exception as e:
                    print(f"[seamless_edge] skipped: {e}", flush=True)

        # ===== 8) Скип SD по флагу =====
        if os.getenv("SKIP_INPAINT", "0") == "1":
            final_bgr = init
            if not self.disable_swap and self.swap_mode == "after":
                print("[faceswap] mode=after", flush=True)
                try:
                    final_bgr = self.swapper.swap(final_bgr, face_bgr, **swap_kwargs)
                except Exception as e: print(f"[faceswap][after][skip_inpaint] failed: {e}", flush=True)
            final_bgr = add_fine_grain(final_bgr, strength=float(os.getenv("GRAIN_STRENGTH","0.012")))
            print(f"[timing] total={time.time()-t0:.2f}s", flush=True)
            return final_bgr

        # ===== 8.7) CPU-fallback =====
        use_cpu = os.getenv("USE_CPU","0") == "1"
        force_cv = os.getenv("FALLBACK_INPAINT","").lower() in {"1","cv"}
        if use_cpu or force_cv:
            dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            small_mask = cv2.dilate(mask_edge, dil, 1)
            inpainted = cv2.inpaint(init, small_mask, 3, cv2.INPAINT_NS)
            final_bgr = inpainted
            if not self.disable_swap and self.swap_mode == "after":
                print("[faceswap] mode=after", flush=True)
                try:
                    final_bgr = self.swapper.swap(final_bgr, face_bgr, **swap_kwargs)
                except Exception as e: print(f"[faceswap][after][cv_fallback] {e}", flush=True)
            final_bgr = add_fine_grain(final_bgr, strength=float(os.getenv("GRAIN_STRENGTH","0.012")))
            print(f"[timing] total={time.time()-t0:.2f}s", flush=True)
            return final_bgr

        # ===== 9) SD Inpaint на патче =====
        _r8 = lambda x: (x+7)//8*8
        pad_px = max(8, int(float(os.getenv("INPAINT_PAD_PCT","0.06")) * max(H, W)))
        x0,y0,x1,y1 = bbox_from_mask(mask_edge, pad=pad_px)
        patch_init  = init[y0:y1, x0:x1]
        patch_mask  = mask_edge[y0:y1, x0:x1]

        long_side = int(os.getenv("INPAINT_LONG","1200"))
        ph, pw = patch_init.shape[:2]
        scale = min(1.0, float(long_side) / max(ph, pw))
        w_s, h_s = max(64, _r8(int(pw*scale))), max(64, _r8(int(ph*scale)))
        init_s = cv2.resize(patch_init, (w_s, h_s), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(patch_mask, (w_s, h_s), interpolation=cv2.INTER_NEAREST)

        grow = int(os.getenv("MASK_GROW","3"))
        if grow > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(grow*2+1,grow*2+1))
            mask_s = cv2.dilate(mask_s, k, 1)

        steps    = int(os.getenv("STEPS","18"))
        guidance = float(os.getenv("GUIDANCE","4.5"))
        strength = float(os.getenv("STRENGTH","0.48"))

        out_img = self.inpainter.inpaint(
            init_rgb=pil_from_bgr(init_s),
            mask_rgb=Image.fromarray(mask_s),
            prompt=prompt,
            negative=self.neg_prompt,
            steps=steps, guidance=guidance, strength=strength
        )
        out_bgr_s = bgr_from_pil(out_img)
        out_bgr   = cv2.resize(out_bgr_s, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # вставляем патч назад
        out_full = init.copy()
        out_full[y0:y1, x0:x1] = out_bgr
        out_bgr = out_full

        # подпитка ВЧ
        out_bgr = highpass_detail(out_bgr, sigma=0.9, gain=0.22)

        # альфа и её уточнение
        m = (mask_edge.astype(np.float32) / 255.0)
        m = refine_alpha_with_edges(selfie_bgr, m, iters=int(os.getenv("EDGE_SNAP_ITERS","1")))
        m3 = m[:, :, None]
        final_bgr = (out_bgr.astype(np.float32)*m3 + init.astype(np.float32)*(1-m3)).astype(np.uint8)

        # реинжект микротекстуры кожи
        if float(os.getenv("FINGER_DETAIL_GAIN","0.35")) > 0:
            final_bgr = texture_reinject(
                final_bgr, selfie_bgr, m,
                sigma=float(os.getenv("FINGER_DETAIL_SIGMA","1.1")),
                gain=float(os.getenv("FINGER_DETAIL_GAIN","0.35"))
            )

        # faceswap (после), если нужно
        if not self.disable_swap and self.swap_mode == "after":
            print("[faceswap] mode=after", flush=True)
            try:
                final_bgr = self.swapper.swap(final_bgr, face_bgr, **swap_kwargs)
            except Exception as e:
                print(f"[faceswap][after] failed: {e}", flush=True)

        # отладка
        if os.getenv("SAVE_DEBUG","0") == "1":
            try:
                cv2.imwrite("/tmp/debug_mask_poly.png", mask_poly)
                cv2.imwrite("/tmp/debug_edge_grad.png", (edge_grad*255).astype(np.uint8))
                cv2.imwrite("/tmp/debug_init.jpg", init)
                cv2.imwrite("/tmp/debug_mask_edge.png", mask_edge)
                cv2.imwrite("/tmp/debug_final.jpg", final_bgr)
            except Exception as e:
                print(f"[debug_save] failed: {e}", flush=True)

        final_bgr = add_fine_grain(final_bgr, strength=float(os.getenv("GRAIN_STRENGTH","0.012")))
        print(f"[timing] total={time.time()-t0:.2f}s", flush=True)
        return final_bgr
