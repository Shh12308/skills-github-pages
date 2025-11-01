# main.py
"""
High-performance image-generation API using Hugging Face Diffusers (Stable Diffusion).
Endpoints:
 - POST /generate : generate an image from a prompt
 - GET  /health   : simple health check

Environment variables:
 - MODEL_ID (default: "runwayml/stable-diffusion-v1-5") or a SDXL checkpoint id
 - DEVICE (cpu|cuda) default: "cuda" if available else "cpu"
 - MAX_CONCURRENCY (int) max parallel generation requests (default 1)
 - OUTPUT_DIR (str) where to write generated files (default "./outputs")
 - CACHE_MAXSIZE (int) LRU cache size (default 256)
 - CACHE_TTL (int) seconds for cache TTL (default 3600)

Run:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Important:
 - Use a GPU for best performance.
 - If using SDXL or other big models, adjust memory and device settings.
"""

import os
import io
import base64
import asyncio
import hashlib
import time
from typing import Optional, Dict, Any
from functools import partial

from fastapi import FastAPI, HTTPException
from fastapi import Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from cachetools import TTLCache, cached
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

# Diffusers imports
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)

# Configuration from environment
MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "256"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Device
if os.getenv("DEVICE"):
    DEVICE = os.getenv("DEVICE")
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="High-End Image Gen API", version="1.0")

# Limit concurrent generation tasks
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# Thread pool for blocking generation calls
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)

# Simple LRU TTL cache for generated outputs keyed by prompt + params
result_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)

# Model pipeline (will be loaded on startup)
PIPELINE = None

# Input schema
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    width: Optional[int] = Field(512, ge=64, le=2048)
    height: Optional[int] = Field(512, ge=64, le=2048)
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=30.0)
    num_inference_steps: Optional[int] = Field(20, ge=1, le=200)
    seed: Optional[int] = Field(None, description="RNG seed (int)")
    upscale: Optional[int] = Field(1, description="Integer upscale factor (1=no upscale, 2,4)")
    format: Optional[str] = Field("b64", description="Response format: 'b64' or 'file'")

async def load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    # Load pipeline with recommended scheduler
    model_id = MODEL_ID
    device = DEVICE
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    # Load model (this can take time; running on startup is preferred)
    try:
        # Use DPMSolverMultistepScheduler for faster generation/quality
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # optionally integrate safety checker
            revision="fp16" if torch_dtype == torch.float16 else None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # Move to device
        pipe = pipe.to(device)
        # Enable attention slicing for memory savings
        pipe.enable_attention_slicing()
        # Enable xformers memory efficient attention if available for speed and VRAM reduction
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        PIPELINE = pipe
        return PIPELINE
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline: {e}")

def prompt_cache_key(params: Dict[str, Any]) -> str:
    # Deterministic cache key for prompt+params
    key_src = "|".join(f"{k}={params.get(k,'')}" for k in sorted(params.keys()))
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")

def upscale_image(img: Image.Image, factor: int) -> Image.Image:
    if factor <= 1:
        return img
    w, h = img.size
    new_size = (w * factor, h * factor)
    # Use high-quality Lanczos resampling for good results without external libs
    return img.resize(new_size, Image.LANCZOS)

def run_generation_sync(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
    device: str,
) -> Image.Image:
    """
    Blocking generation call that is executed inside a thread to avoid blocking asyncio loop.
    Returns a PIL Image.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Some pipelines expect (height, width) arguments differently across versions.
    # We'll pass height/width explicitly if supported.
    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        image = result.images[0]
        return image
    except TypeError:
        # Fallback for older/newer signature differences
        result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        # If size mismatch, resize to requested dims (may degrade)
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
        return image

@app.on_event("startup")
async def startup_event():
    # Load pipeline at startup
    app.state.pipeline = await load_pipeline()

@app.get("/health")
async def health():
    ready = app.state.pipeline is not None
    return {"status": "ok", "ready": ready, "device": DEVICE, "model": MODEL_ID}

@app.post("/generate")
async def generate(req: GenerateRequest):
    # Validate and set defaults
    width = req.width or 512
    height = req.height or 512
    guidance_scale = req.guidance_scale or 7.5
    num_inference_steps = req.num_inference_steps or 20
    seed = req.seed
    upscale_factor = max(1, int(req.upscale or 1))
    fmt = (req.format or "b64").lower()
    if fmt not in ("b64", "file"):
        raise HTTPException(status_code=400, detail="format must be 'b64' or 'file'")

    # Build cache key
    params = {
        "prompt": req.prompt,
        "negative": req.negative_prompt or "",
        "width": width,
        "height": height,
        "guidance": guidance_scale,
        "steps": num_inference_steps,
        "seed": seed or "",
        "upscale": upscale_factor,
    }
    cache_key = prompt_cache_key(params)
    if cache_key in result_cache:
        cached_path, cached_b64 = result_cache[cache_key]
        # Return cached result quickly
        if fmt == "b64":
            return {"cached": True, "image_b64": cached_b64}
        else:
            return {"cached": True, "file": cached_path}

    # Acquire concurrency slot
    async with semaphore:
        pipeline = app.state.pipeline
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")

        # Run generation in thread pool
        loop = asyncio.get_running_loop()
        try:
            image: Image.Image = await loop.run_in_executor(
                thread_pool,
                partial(
                    run_generation_sync,
                    pipeline,
                    req.prompt,
                    req.negative_prompt,
                    width,
                    height,
                    guidance_scale,
                    num_inference_steps,
                    seed,
                    DEVICE,
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {e}")

        # Optionally upscale using high-quality resampling
        if upscale_factor > 1:
            try:
                image = await loop.run_in_executor(thread_pool, partial(upscale_image, image, upscale_factor))
            except Exception as e:
                # Upscale failure fallback: continue with original image
                pass

        # Convert to base64 and optionally save to file
        b64 = pil_to_base64_png(image)
        timestamp = int(time.time() * 1000)
        filename = f"gen_{timestamp}_{cache_key[:8]}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        # Save in background thread
        def _save_image(img, p):
            img.save(p, format="PNG", optimize=True)

        await loop.run_in_executor(thread_pool, partial(_save_image, image, path))

        # Store in cache
        result_cache[cache_key] = (path, b64)

        if fmt == "b64":
            return {"cached": False, "image_b64": b64, "seed": seed, "path": path}
        else:
            return {"cached": False, "file": path, "seed": seed}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
