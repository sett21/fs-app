import os
from tkinter import Image
from typing import Optional
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
import numpy as torch

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
        try:
            self.pipe = _load(True)
        except Exception:
            self.pipe = _load(False)

        try: self.pipe.enable_attention_slicing()
        except: pass
        try: self.pipe.enable_vae_tiling()
        except: pass

        # GPU/CPU режим без конфликтов offload
        if device == "cpu":
            try: self.pipe.enable_sequential_cpu_offload()
            except: pass
        else:
            self.pipe = self.pipe.to("cuda")
        self.device = device

    @torch.inference_mode()
    def inpaint(self, init_rgb: Image.Image, mask_rgb: Image.Image,
                prompt: Optional[str]=None, negative: Optional[str]=None,
                steps: int=24, guidance: float=5.0, strength: float=0.5) -> Image.Image:
        if not prompt or not str(prompt).strip():
            prompt = os.getenv(
                "DEFAULT_PROMPT",
                "photorealistic selfie; person holding a postcard; keep postcard unchanged; "
                "natural skin texture; soft contact shadows; realistic fingers, no blur"
            )
        if negative is None:
            negative = os.getenv(
                "NEG_PROMPT",
                "altered postcard, changed text, hdr halos, glow edges, extra fingers, "
                "deformed hands, warped text, logo distortion, oversharpen, blur, artifacts"
            )
        if mask_rgb.mode != "L":
            mask_rgb = mask_rgb.convert("L")
        return self.pipe(
            prompt=prompt, negative_prompt=negative,
            image=init_rgb, mask_image=mask_rgb,
            guidance_scale=guidance, num_inference_steps=steps, strength=strength
        ).images[0]