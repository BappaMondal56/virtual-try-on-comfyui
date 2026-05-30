import os
import sys
import random
sys.path.append(os.getcwd())

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from tryon_generation import find_path, _ensure_models_loaded
from tryon_generation import generate_tryon_image
import threading

app = FastAPI(title="TryOn API", description="Virtual try-on using ComfyUI workflow", version="1.0")

# Generation lock to serialize heavy GPU/CPU work
gen_lock = threading.Lock()


@app.on_event("startup")
def startup():
    # Preload models and custom nodes once at startup to avoid doing this during requests
    try:
        _ensure_models_loaded()
    except Exception as e:
        print("Failed to preload models on startup:", e, file=sys.stderr)

COMFY_PATH = find_path("ComfyUI")
if COMFY_PATH is None:
    raise RuntimeError("ComfyUI directory not found. Make sure it exists in the path.")

OUTPUT_DIR = os.path.join(COMFY_PATH, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")


class TryOnRequest(BaseModel):
    model_filename: str = Field(..., description="Filename or path to the model image (person). If only a filename is provided, file is looked up in `ComfyUI/input`.")
    outfit_filename: str = Field(..., description="Filename or path to the outfit image (clothing). If only a filename is provided, file is looked up in `ComfyUI/input`.")
    steps: Optional[int] = None
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


class GenerateResponse(BaseModel):
    status: str
    prompt_used: str
    images: list[str]


@app.post("/tryon", response_model=GenerateResponse, tags=["TryOn"])
def tryon_generate(data: TryOnRequest):
    try:
        base_input = os.path.join(os.getcwd(), "ComfyUI", "input")
        model_path = data.model_filename if os.path.exists(data.model_filename) else os.path.join(base_input, data.model_filename)
        outfit_path = data.outfit_filename if os.path.exists(data.outfit_filename) else os.path.join(base_input, data.outfit_filename)

        # Serialize generation to avoid concurrent model access
        with gen_lock:
            result = generate_tryon_image(
            model_path=model_path,
            outfit_path=outfit_path,
            filename_prefix="tryon",
            steps=data.steps,
            cfg=data.cfg,
            width=data.width,
            height=data.height,
            )

        filenames = []
        for item in result.get("ui", {}).get("images", result.get("images", [])):
            subfolder = item.get("subfolder", "")
            filename = item.get("filename", "")
            rel = os.path.join(subfolder, filename).lstrip("/") if subfolder else filename
            filenames.append(f"/images/{rel}")

        return GenerateResponse(status="success", prompt_used="tryon", images=filenames)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting TryOn API")
    uvicorn.run(app, host="0.0.0.0", port=8001)
