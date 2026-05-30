import os
import random
import sys
import torch
from typing import Sequence, Mapping, Any, Union

# Protect against ComfyUI argparse conflict
_saved_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]


has_manager = False


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        return os.path.join(path, name)
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        manager_path = os.path.join(comfyui_path, "custom_nodes", "ComfyUI-Manager", "glob")
        if os.path.isdir(manager_path) and os.listdir(manager_path):
            sys.path.append(manager_path)
            global has_manager
            has_manager = True


def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except Exception:
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)


# restore argv now that we know ComfyUI path may be discovered later
sys.argv = _saved_argv


# Model loading state
_models_loaded = False
_custom_path_added = False
_custom_nodes_imported = False

CLIP = None
VAE = None
UNET = None
LORA = None

MODEL_CONFIG = {
    "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "clip_type": "qwen_image",
    "vae_name": "qwen_image_vae.safetensors",
    "unet_name": "Qwen-Image-Edit-2509-Q3_K_M.gguf",
    "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
    "lora_strength": 1,
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "cfg": 1,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": 1,
    "filename_prefix": "tryon",
}


def import_custom_nodes() -> None:
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            pass
        else:
            if hasattr(manager, "get_config"):
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception:
                    pass

    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)
        await init_extra_nodes(init_custom_nodes=True)

    loop.run_until_complete(inner())


def _ensure_models_loaded():
    global _models_loaded, _custom_path_added, _custom_nodes_imported
    global CLIP, VAE, UNET, LORA

    if _models_loaded:
        return

    if not _custom_path_added:
        add_comfyui_directory_to_sys_path()
        add_extra_model_paths()
        _custom_path_added = True

    if not _custom_nodes_imported:
        import_custom_nodes()
        _custom_nodes_imported = True

    from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode():
        # Try GGUF clip loader first, fallback to CLIPLoader
        try:
            clip_loader = NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
            if clip_loader is None:
                clip_loader = NODE_CLASS_MAPPINGS["CLIPLoader"]
            clip_inst = clip_loader()
            CLIP = clip_inst.load_clip(clip_name=MODEL_CONFIG["clip_name"], type=MODEL_CONFIG["clip_type"]) if hasattr(clip_inst, "load_clip") else clip_inst
        except Exception:
            CLIP = None

        vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]()
        VAE = vae_loader.load_vae(vae_name=MODEL_CONFIG["vae_name"]) if hasattr(vae_loader, "load_vae") else vae_loader

        unet_loader = NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF", NODE_CLASS_MAPPINGS.get("UnetLoader"))()
        UNET = unet_loader.load_unet(unet_name=MODEL_CONFIG["unet_name"]) if hasattr(unet_loader, "load_unet") else unet_loader

        # Optionally load LoRA into a wrapper if available
        if "LoraLoaderModelOnly" in NODE_CLASS_MAPPINGS and MODEL_CONFIG.get("lora_name"):
            try:
                lora_loader_cls = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]
                lora_loader = lora_loader_cls()
                if hasattr(lora_loader, "load_lora_model_only"):
                    LORA = lora_loader.load_lora_model_only(
                        lora_name=MODEL_CONFIG["lora_name"],
                        strength_model=MODEL_CONFIG.get("lora_strength", 1),
                        model=get_value_at_index(UNET, 0),
                    )
                elif hasattr(lora_loader, "load_lora"):
                    LORA = lora_loader.load_lora(lora_name=MODEL_CONFIG["lora_name"], strength_model=MODEL_CONFIG.get("lora_strength", 1), model=get_value_at_index(UNET, 0))
                elif hasattr(lora_loader, "load"):
                    LORA = lora_loader.load(lora_name=MODEL_CONFIG["lora_name"], strength_model=MODEL_CONFIG.get("lora_strength", 1), model=get_value_at_index(UNET, 0))
                else:
                    raise RuntimeError("LoraLoaderModelOnly node has no supported load method")
            except Exception:
                LORA = None
        else:
            LORA = None

    _models_loaded = True


def generate_tryon_image(
    model_path: str,
    outfit_path: str,
    filename_prefix: str = "tryon",
    steps: int = None,
    cfg: float = None,
    width: int = None,
    height: int = None,
) -> dict:
    """Generate a virtual try-on image by stitching the model + outfit images
    and running the Qwen image-edit workflow.

    Paths may be absolute or relative to `ComfyUI/input`.
    Returns the same dict structure as other generators (contains saved image info).
    """
    _ensure_models_loaded()

    from nodes import NODE_CLASS_MAPPINGS

    _steps = steps if steps is not None else MODEL_CONFIG["steps"]
    _cfg = cfg if cfg is not None else MODEL_CONFIG["cfg"]
    _width = width if width is not None else MODEL_CONFIG["width"]
    _height = height if height is not None else MODEL_CONFIG["height"]

    base_input = os.path.join(os.getcwd(), "ComfyUI", "input")
    model_file = model_path if os.path.exists(model_path) else os.path.join(base_input, model_path)
    outfit_file = outfit_path if os.path.exists(outfit_path) else os.path.join(base_input, outfit_path)

    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        load_model = loadimage.load_image(image=model_file)
        load_outfit = loadimage.load_image(image=outfit_file)

        imagestitch = NODE_CLASS_MAPPINGS["ImageStitch"]()
        stitched = imagestitch.stitch(
            direction="right",
            match_image_size=True,
            spacing_width=0,
            spacing_color="white",
            image1=get_value_at_index(load_model, 0),
            image2=get_value_at_index(load_outfit, 0),
        )

        imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
        try:
            scaled = imagescaletototalpixels.EXECUTE_NORMALIZED(
                upscale_method="lanczos",
                megapixels=1,
                resolution_steps=1,
                image=get_value_at_index(stitched, 0),
            )
        except TypeError:
            scaled = imagescaletototalpixels.EXECUTE_NORMALIZED(
                upscale_method="lanczos",
                megapixels=1,
                image=get_value_at_index(stitched, 0),
            )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        _ = vaeencode.encode(
            pixels=get_value_at_index(scaled, 0),
            vae=get_value_at_index(VAE, 0),
        )

        emptylatent = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        empty_latent = emptylatent.generate(width=_width, height=_height, batch_size=1)

        # Prepare model for sampling (use loaded UNET wrapper)
        model_for_sampling = get_value_at_index(UNET, 0)

        # If a LoRA wrapper was loaded, prefer its output as the sampling model
        if LORA is not None:
            try:
                model_for_sampling = get_value_at_index(LORA, 0)
            except Exception:
                model_for_sampling = LORA

        # Optional aura + cfgnorm (if nodes exist)
        if "ModelSamplingAuraFlow" in NODE_CLASS_MAPPINGS and "CFGNorm" in NODE_CLASS_MAPPINGS:
            aura = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
            aura_patched = aura.patch_aura(shift=3, model=model_for_sampling)
            cfgn = NODE_CLASS_MAPPINGS["CFGNorm"]()
            model_for_sampling = cfgn.EXECUTE_NORMALIZED(strength=1, model=get_value_at_index(aura_patched, 0))

        textencode = NODE_CLASS_MAPPINGS["TextEncodeQwenImageEdit"]()
        pos = textencode.EXECUTE_NORMALIZED(
            prompt=(
                "show the model wearing the uploaded outfit in place of the outfit on the model's image, "
                "no part of old outfit should be present. Keep the model's face, pose, and background unchanged. "
                "The new clothing should fully replace the old one of the model and fit naturally."
            ),
            clip=get_value_at_index(CLIP, 0),
            vae=get_value_at_index(VAE, 0),
            image=get_value_at_index(scaled, 0),
        )

        neg = textencode.EXECUTE_NORMALIZED(
            prompt="",
            clip=get_value_at_index(CLIP, 0),
            vae=get_value_at_index(VAE, 0),
            image=get_value_at_index(scaled, 0),
        )

        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        sampled = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=_steps,
            cfg=_cfg,
            sampler_name=MODEL_CONFIG["sampler_name"],
            scheduler=MODEL_CONFIG["scheduler"],
            denoise=MODEL_CONFIG["denoise"],
            model=get_value_at_index(model_for_sampling, 0) if isinstance(model_for_sampling, (list, tuple)) else model_for_sampling,
            positive=get_value_at_index(pos, 0),
            negative=get_value_at_index(neg, 0),
            latent_image=get_value_at_index(empty_latent, 0),
        )

        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded = vaedecode.decode(samples=get_value_at_index(sampled, 0), vae=get_value_at_index(VAE, 0))

        saver = NODE_CLASS_MAPPINGS["SaveImage"]()
        result = saver.save_images(filename_prefix=filename_prefix, images=get_value_at_index(decoded, 0))

    return result
