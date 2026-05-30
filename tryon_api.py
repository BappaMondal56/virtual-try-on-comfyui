import os
import sys
import random
sys.path.append(os.getcwd())

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List

from tryon_generation import find_path, _ensure_models_loaded, generate_tryon_image
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


class GenerateResponse(BaseModel):
    status: str
    prompt_used: str
    images: List[str]


def _run_generation_sync(model_path: str, outfit_path: str, filename_prefix: str = "tryon"):
    # run the (heavy) sync generation under a thread lock
    with gen_lock:
        return generate_tryon_image(model_path=model_path, outfit_path=outfit_path, filename_prefix=filename_prefix)


@app.post("/tryon", response_model=GenerateResponse, tags=["TryOn"])
async def tryon_generate(model_filename: str = Form(...), outfit_file: UploadFile = File(...)):
    """
    Accepts a model image filename (must exist in `ComfyUI/input`) and an uploaded outfit image.
    The uploaded outfit is saved into `ComfyUI/input` and the Qwen-based tryon workflow runs.
    Only `model_filename` and the uploaded file are required; no other parameters are accepted.
    """
    try:
        base_input = os.path.join(os.getcwd(), "ComfyUI", "input")
        os.makedirs(base_input, exist_ok=True)

        # Resolve model path from input folder
        model_path = model_filename if os.path.isabs(model_filename) and os.path.exists(model_filename) else os.path.join(base_input, model_filename)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail=f"Model image not found: {model_filename}")

        # Save uploaded outfit into ComfyUI/input
        outfit_name = outfit_file.filename or f"uploaded_outfit_{random.randint(1000,9999)}.png"
        dest_path = os.path.join(base_input, outfit_name)
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(outfit_name)
            dest_path = os.path.join(base_input, f"{name}_{random.randint(1000,9999)}{ext}")

        contents = await outfit_file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)

        # Run generation in a thread pool to avoid blocking the event loop
        result = await run_in_threadpool(_run_generation_sync, model_path, dest_path, "tryon")

        filenames: List[str] = []
        for item in result.get("ui", {}).get("images", result.get("images", [])):
            subfolder = item.get("subfolder", "")
            filename = item.get("filename", "")
            rel = os.path.join(subfolder, filename).lstrip("/") if subfolder else filename
            filenames.append(f"/images/{rel}")

        return GenerateResponse(status="success", prompt_used="tryon", images=filenames)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting TryOn API")
    uvicorn.run(app, host="0.0.0.0", port=8001)