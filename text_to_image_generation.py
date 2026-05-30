import os
import random
import sys
import json
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("root").setLevel(logging.ERROR)

# ==============================
# Utility Functions
# ==============================

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
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)

        manager_path = os.path.join(
            comfyui_path, "custom_nodes", "ComfyUI-Manager", "glob"
        )
        if os.path.isdir(manager_path) and os.listdir(manager_path):
            sys.path.append(manager_path)
            global has_manager
            has_manager = True

        import __main__
        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    from comfy.options import enable_args_parsing
    enable_args_parsing()
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            print("Could not import manager_core, proceeding without it.")
            return
        else:
            if hasattr(manager, "get_config"):
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception as e:
                    print("Failed to patch manager_core.get_config:", e)

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


# ==============================
# Prompt Builder (Dynamic)
# ==============================

BODY_MAP = {
    "fit": "athletic toned body",
    "slim": "slender petite body",
    "curvy": "curvy hourglass figure",
    "chubby": "soft plus-size body",
    # add more if needed
}

HAIR_MAP = {
    "long": "long flowing",
    "short": "short sleek",
    "ponytail": "high ponytail",
    "braided": "thick braids",
    "wavy": "wavy cascading",
    "bangs": "bangs framing face",
    "buns": "messy hair buns",
    # add more
}

BREAST_SIZE_MAP = {
    "flat":       "completely flat chest, tiny barely-there A-cup breasts, minimal breast tissue, small flush nipples almost invisible, narrow flat areolas, no cleavage or projection whatsoever",
    "small":      "small perky B-cup breasts, modest gentle curve, small erect pink nipples, compact areolas, light natural shape without heaviness",
    "medium":     "medium C-cup breasts, natural rounded shape, erect pink nipples, moderately wide areolas, soft natural cleavage",
    "large":      "large full D-cup breasts, prominent natural pendulous shape resting heavily, erect prominent pink nipples, wide textured areolas, noticeable deep cleavage and soft undercurve",
    "xl":         "massive enormous F-cup breasts, huge pendulous heavy shape dominating the frame, very prominent erect pink nipples, extremely wide detailed areolas, dramatic deep cleavage and natural sag",
    # add more levels if you want (tiny, huge, gigantic, etc.)
}

BUTT_SIZE_MAP = {  # similar idea — optional
    "small":   "slim hips, small firm butt",
    "medium":  "moderately curved hips, rounded butt",
    "large":   "wide hips, thick plump butt",
    "xl":      "very wide hips, massive thick plush butt",
}


# def build_prompt(
#     gender: str,
#     ethnicity: str,
#     skin_tone: str,
#     hair_style: str,
#     hair_color: str,
#     body_type: str,
#     age: int,
#     breast_size: str = "",
#     butt_size: str = ""
# ) -> str:
#     body_desc = BODY_MAP.get(body_type.lower(), body_type)
#     hair_desc = HAIR_MAP.get(hair_style.lower(), hair_style)

#     extra_features = ""
#     if breast_size:
#         extra_features += f", {breast_size} bust"
#     if butt_size:
#         extra_features += f", {butt_size} hips"

#     prompt = f"""
#     Full body front-facing portrait of a stunning {age} year old {ethnicity} {gender} with {skin_tone} skin, completely nude except for tiny beige thong panties pulled high on her hips, sitting seductively on a velvet chaise at the foot of a grand bed in an erotic luxurious bedroom. She has {hair_desc} in {hair_color} color,"""

#     return " ".join(prompt.split())


def build_prompt(
    gender: str = "woman",
    ethnicity: str = "Indian",
    eye_color: str = "brown",
    skin_tone: str = "lightly tanned fair",
    hair_style: str = "braided",
    hair_color: str = "black",
    body_type: str = "curvy",
    age: int = 23,
    breast_size: str = "",   # "flat", "small", "medium", "large", "xl", ""
    butt_size: str = ""      # "small", "medium", "large", "xl", ""
) -> str:
    body_desc = BODY_MAP.get(body_type.lower(), body_type.lower())
    hair_desc = HAIR_MAP.get(hair_style.lower(), hair_style.lower())

    # Breast description — use mapped detailed version or fallback
    breast_desc = BREAST_SIZE_MAP.get(breast_size.lower(), "")
    if breast_desc:
        breast_desc = f", {breast_desc}"

    # Butt description
    butt_desc = BUTT_SIZE_MAP.get(butt_size.lower(), "")
    if butt_desc:
        butt_desc = f", {butt_desc}"

    # Core erotic bedroom scene (same as before, consistent)
    base_scene = (
        "sitting seductively on luxurious silk bedding in an erotic opulent bedroom, "
        "rumpled crimson satin sheets on massive bed, scattered plush pillows, "
        "sheer dark drapes, warm candlelight and dim sconces creating intimate rim lighting "
        "and glowing skin highlights, volumetric soft fog, moody sensual atmosphere"
    )

    prompt = f"""
Full body front-facing portrait of a stunning {age}-year-old {ethnicity} {gender} with {skin_tone} skin, 
completely nude except for tiny beige thong panties pulled high on her hips, 
{base_scene}. 

She has {hair_desc} {hair_color} hair falling over her shoulders and chest, strands slightly messy with realistic shine and texture. 
Seductive {eye_color} eyes gazing directly at viewer with subtle confident smile, full glossy lips, 
flawless symmetrical face with natural makeup, light freckles on cheeks, shoulders and upper chest.

{body_desc}{breast_desc}{butt_desc}, 
hyper-realistic skin texture with visible pores, subtle sweat sheen, natural body shadows and subsurface scattering.

She sits with legs slightly apart, one hand resting on her thigh, the other braced on the bed, posture relaxed and inviting.

Photorealistic ultra-realistic rendering, hyper-detailed anatomy and skin, masterpiece, best quality, 8K resolution, sharp focus, 
professional erotic boudoir photography style, shot on Sony A1 85mm lens f/1.4, high dynamic range, rich natural warm color grading.
"""

    # Clean up extra spaces
    return " ".join(prompt.split())


# ==============================
# Model Config
# ==============================

MODEL_CONFIG = {
    "clip_name": "Qwen3-4B-Q5_K_M.gguf",
    "clip_type": "lumina2",
    "vae_name": "z_image_turbo_vae.safetensors",
    "unet_name": "z_image_turbo-Q5_K_M.gguf",
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "cfg": 1,
    "sampler_name": "res_multistep",
    "scheduler": "simple",
    "denoise": 1,
    "filename_prefix": "z-image/i",
}

NEGATIVE_PROMPT = "blurry, low quality, distorted face, extra limbs, deformed"

# ==============================
# Load Models ONCE
# ==============================

_models_loaded = False
_custom_path_added = False
_custom_nodes_imported = False

CLIP = None
VAE = None
UNET = None


def _ensure_models_loaded():
    global _models_loaded, _custom_path_added, _custom_nodes_imported
    global CLIP, VAE, UNET

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

    print("Loading ComfyUI models (z-image turbo)...")

    with torch.inference_mode():
        clip_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        CLIP = clip_loader.load_clip(
            clip_name=MODEL_CONFIG["clip_name"],
            type=MODEL_CONFIG["clip_type"],
        )

        vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]()
        VAE = vae_loader.load_vae(vae_name=MODEL_CONFIG["vae_name"])

        unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        UNET = unet_loader.load_unet(unet_name=MODEL_CONFIG["unet_name"])

    _models_loaded = True
    print("Models loaded successfully.")


# ==============================
# Image Generation Function
# ==============================

def generate_model_image(
    prompt_text: str,
    negative_prompt: str = NEGATIVE_PROMPT,
    seed: int = None,
    steps: int = None,
    cfg: float = None,
    width: int = None,
    height: int = None,
    filename_prefix: str = None,
) -> dict:
    """
    Generate an image using z-image-turbo pipeline.

    Returns a dict with 'images' key containing list of {filename, subfolder, type}.
    """
    _ensure_models_loaded()

    from nodes import NODE_CLASS_MAPPINGS

    _seed = seed if seed is not None else random.randint(1, 2**64)
    _steps = steps if steps is not None else MODEL_CONFIG["steps"]
    _cfg = cfg if cfg is not None else MODEL_CONFIG["cfg"]
    _width = width if width is not None else MODEL_CONFIG["width"]
    _height = height if height is not None else MODEL_CONFIG["height"]
    _prefix = filename_prefix if filename_prefix is not None else MODEL_CONFIG["filename_prefix"]

    with torch.inference_mode():
        # Encode positive prompt
        clip_encoder = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        positive = clip_encoder.encode(
            text=prompt_text,
            clip=get_value_at_index(CLIP, 0),
        )

        # Zero out for negative conditioning (as in original workflow)
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        negative = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(positive, 0)
        )

        # Empty latent
        latent_node = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
        latent = latent_node.EXECUTE_NORMALIZED(
            width=_width,
            height=_height,
            batch_size=1,
        )

        # Sample
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        samples = ksampler.sample(
            seed=_seed,
            steps=_steps,
            cfg=_cfg,
            sampler_name=MODEL_CONFIG["sampler_name"],
            scheduler=MODEL_CONFIG["scheduler"],
            denoise=MODEL_CONFIG["denoise"],
            model=get_value_at_index(UNET, 0),
            positive=get_value_at_index(positive, 0),
            negative=get_value_at_index(negative, 0),
            latent_image=get_value_at_index(latent, 0),
        )

        # Decode
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded = vaedecode.decode(
            samples=get_value_at_index(samples, 0),
            vae=get_value_at_index(VAE, 0),
        )

        # Save
        saver = NODE_CLASS_MAPPINGS["SaveImage"]()
        result = saver.save_images(
            filename_prefix=_prefix,
            images=get_value_at_index(decoded, 0),
        )

    return result
