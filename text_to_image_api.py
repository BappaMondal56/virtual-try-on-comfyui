import os
import sys
import random
sys.path.append(os.getcwd())

import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("root").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from z_image_generation import build_prompt, generate_model_image, find_path

# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Z-Image Turbo Generator API",
    description="Generate photorealistic AI model images using z-image-turbo + ComfyUI",
    version="2.0",
)

# =========================
# Locate ComfyUI Output Dir
# =========================

COMFY_PATH = find_path("ComfyUI")
if COMFY_PATH is None:
    raise RuntimeError("ComfyUI directory not found. Make sure it exists in the path.")

OUTPUT_DIR = os.path.join(COMFY_PATH, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")


# =========================
# Internal generation defaults (not exposed in API)
# =========================

GEN_STEPS   = 8
GEN_CFG     = 1.0
GEN_WIDTH   = 1024
GEN_HEIGHT  = 1024


# =========================
# Request / Response Schema
# =========================

class ModelRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "gender": "female",
        "ethnicity": "indian",
        "eye_color": "brown",
        "skin_tone": "fair",
        "hair_style": "ponytail",
        "hair_color": "black",
        "body_type": "fit",
        "age": 23,
        "breast_size": "medium",
        "butt_size": "medium",
    }}}

    gender: str
    ethnicity: str
    eye_color: str
    skin_tone: str
    hair_style: str = Field(description="Options: long, short, ponytail, braided, wavy, bangs, buns")
    hair_color: str
    body_type: str = Field(description="Options: fit, slim, curvy, chubby")
    age: int = Field(ge=18, le=80)
    breast_size: str = ""
    butt_size: str = ""


class GenerateResponse(BaseModel):
    status: str
    prompt_used: str
    images: list[str]


# =========================
# Health Check
# =========================

@app.get("/", tags=["Health"])
def home():
    return {
        "status": "Z-Image Turbo API running",
        "docs": "/docs",
        "generate": "POST /generate",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


# =========================
# Generate Endpoint
# =========================

@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
def generate_model(data: ModelRequest):
    """
    Generate a photorealistic AI model image.
    Returns image URLs accessible via /images/<filename>.
    """
    try:
        prompt = build_prompt(
            gender=data.gender,
            ethnicity=data.ethnicity,
            eye_color=data.eye_color,
            skin_tone=data.skin_tone,
            hair_style=data.hair_style,
            hair_color=data.hair_color,
            body_type=data.body_type,
            age=data.age,
            breast_size=data.breast_size,
            butt_size=data.butt_size,
        )

        result = generate_model_image(
            prompt_text=prompt,
            seed=random.randint(1, 2**64),
            steps=GEN_STEPS,
            cfg=GEN_CFG,
            width=GEN_WIDTH,
            height=GEN_HEIGHT,
        )

        filenames = []
        for item in result.get("ui", {}).get("images", result.get("images", [])):
            subfolder = item.get("subfolder", "")
            filename = item.get("filename", "")
            rel = os.path.join(subfolder, filename).lstrip("/") if subfolder else filename
            filenames.append(f"/images/{rel}")

        return GenerateResponse(
            status="success",
            prompt_used=prompt,
            images=filenames,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Prompt Preview (no image)
# =========================

@app.post("/preview-prompt", tags=["Utility"])
def preview_prompt(data: ModelRequest):
    """Returns the prompt that would be generated without running inference."""
    prompt = build_prompt(
        gender=data.gender,
        ethnicity=data.ethnicity,
        eye_color=data.eye_color,
        skin_tone=data.skin_tone,
        hair_style=data.hair_style,
        hair_color=data.hair_color,
        body_type=data.body_type,
        age=data.age,
        breast_size=data.breast_size,
        butt_size=data.butt_size,
    )
    return {"prompt": prompt}


# =========================
# List Available Options
# =========================

@app.get("/options", tags=["Utility"])
def list_options():
    """Returns all supported field values."""
    return {
        "body_type":   ["fit", "slim", "curvy", "chubby"],
        "hair_style":  ["long", "short", "ponytail", "braided", "wavy", "bangs", "buns"],
        "eye_color":   ["brown", "blue", "green", "hazel"],
        "breast_size": ["flat", "small", "medium", "large", "xl"],
        "butt_size":   ["small", "medium", "large", "xl"],
        "skin_tone":   ["light", "fair", "lightly tanned", "tanned", "dark"],
    }


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Z-Image Turbo API")
    print("📄 Docs: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )