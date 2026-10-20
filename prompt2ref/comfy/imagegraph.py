"""Reference-conditioned *image* backends, as ComfyUI API graphs.

The video side has one model and a shipped template to edit. Images don't: the
useful reference-conditioned image models each wire their references in their
own way, so each backend here is a small function that returns a flat
API-format graph, plus a description of what it needs and what it can do.

    qwen-image-edit   Qwen-Image-Edit 2511. References ride in TextEncodeQwenImage-
                      EditPlus as up to three conditioning images; the same node
                      does editing, with the edited frame taking slot 1.
    flux2-klein       Flux.2 Klein. References are VAE-encoded and chained onto
                      the positive conditioning as ReferenceLatents, so the slot
                      count is only bounded by VRAM.
    qwen-image        Qwen-Image 2512, plain text-to-image. No image conditioning
                      at all - the references only inform the written prompt.
                      Here as the "same prompt, no reference wiring" control.

Model files are matched by substring against whatever the target ComfyUI
actually has, so renamed or requantised checkpoints still resolve.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .template import solve_resolution

INPUT_SUBDIR = "prompt2ref"
SAVE_PREFIX = "prompt2ref/p2r_image"


class ModelMissing(RuntimeError):
    pass


@dataclass
class ImageBackend:
    key: str
    label: str
    ref_slots: int                 # reference images the model can be conditioned on
    supports_edit: bool
    tagged_refs: bool              # does the prompt address references as "Picture N"?
    unet: tuple[str, ...]
    clip: tuple[str, ...]
    clip_type: str
    vae: tuple[str, ...]
    turbo_lora: tuple[str, ...] = ()
    steps: int = 20
    cfg: float = 2.5
    turbo_steps: int = 4
    turbo_cfg: float = 1.0
    shift: float | None = None     # ModelSamplingAuraFlow
    guidance: float | None = None  # FluxGuidance
    sampler: str = "euler"
    scheduler: str = "simple"
    megapixels: float = 1.0
    multiple: int = 16
    notes: str = ""

    def defaults(self, turbo: bool) -> tuple[int, float]:
        return (self.turbo_steps, self.turbo_cfg) if turbo and self.turbo_lora \
            else (self.steps, self.cfg)

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "ref_slots": self.ref_slots,
            "supports_edit": self.supports_edit, "has_turbo": bool(self.turbo_lora),
            "steps": self.steps, "cfg": self.cfg,
            "turbo_steps": self.turbo_steps, "turbo_cfg": self.turbo_cfg,
            "megapixels": self.megapixels, "notes": self.notes,
        }


BACKENDS: dict[str, ImageBackend] = {}


def _register(b: ImageBackend) -> ImageBackend:
    BACKENDS[b.key] = b
    return b


QWEN_EDIT = _register(ImageBackend(
    key="qwen-image-edit",
    label="Qwen-Image-Edit 2511 (reference generate + edit)",
    ref_slots=3, supports_edit=True, tagged_refs=True,
    unet=("qwen_image_edit", "qwen-image-edit"),
    clip=("qwen_2.5_vl_7b", "qwen2.5_vl_7b"), clip_type="qwen_image",
    vae=("qwen_image_vae",),
    turbo_lora=("qwen-image-edit-2509-lightning-4steps", "qwen-image-edit-lightning-4steps"),
    steps=20, cfg=2.5, turbo_steps=4, turbo_cfg=1.0, shift=3.0,
    notes="three conditioning slots; in edit mode the edited frame takes the first one",
))

FLUX2_KLEIN = _register(ImageBackend(
    key="flux2-klein",
    label="Flux.2 Klein (multi-reference generate + edit)",
    ref_slots=4, supports_edit=True, tagged_refs=False,
    unet=("flux-2-klein", "flux2-klein"),
    # Klein 9B projects three concatenated 4096-wide layers, i.e. plain Qwen3-8B.
    # The 4B encoder loads happily and then dies in the sampler with a 7680 vs
    # 12288 matmul, and "qwen3vl_8b" is the VL variant, not this one.
    clip=("qwen_3_8b", "qwen3_8b"), clip_type="flux2",
    vae=("flux2-vae", "flux2_vae"),
    steps=20, cfg=1.0, guidance=4.0, scheduler="simple",
    notes="references are chained reference latents - no hard slot limit, but each one costs VRAM",
))

QWEN_T2I = _register(ImageBackend(
    key="qwen-image",
    label="Qwen-Image 2512 (text-to-image, no reference conditioning)",
    ref_slots=0, supports_edit=False, tagged_refs=False,
    unet=("qwen-image-2512", "qwen_image_2512"),
    clip=("qwen_2.5_vl_7b", "qwen2.5_vl_7b"), clip_type="qwen_image",
    vae=("qwen_image_vae",),
    turbo_lora=("wuli-qwen-image-2512-turbo",),
    steps=20, cfg=2.5, turbo_steps=4, turbo_cfg=1.0, shift=3.1,
    notes="the references shape the written prompt only; nothing is fed to the model as an image",
))

DEFAULT_BACKEND = QWEN_EDIT.key


# --- model discovery --------------------------------------------------------

MODEL_DIRS = {"unet": ("diffusion_models", "unet"), "clip": ("text_encoders", "clip"),
              "vae": ("vae",), "loras": ("loras",)}


def list_models(server: str | None = None, comfy_root: Path | None = None) -> dict[str, list[str]]:
    """Every model filename ComfyUI would offer, from the server or from disk."""
    if server:
        try:
            return {
                "unet": _combo(server, "UNETLoader", "unet_name"),
                "clip": _combo(server, "CLIPLoader", "clip_name"),
                "vae": _combo(server, "VAELoader", "vae_name"),
                "loras": _combo(server, "LoraLoaderModelOnly", "lora_name"),
            }
        except Exception:
            pass
    root = Path(comfy_root) if comfy_root else None
    pools: dict[str, list[str]] = {k: [] for k in MODEL_DIRS}
    if root:
        for kind, dirs in MODEL_DIRS.items():
            for d in dirs:
                folder = root / "models" / d
                if folder.is_dir():
                    pools[kind].extend(
                        sorted(f.name for f in folder.rglob("*")
                               if f.suffix in (".safetensors", ".sft", ".gguf")))
    return pools


def _combo(server: str, node: str, field_name: str) -> list[str]:
    with urllib.request.urlopen("%s/object_info/%s" % (server.rstrip("/"), node),
                                timeout=20) as r:
        info = json.load(r)[node]
    return list(info["input"]["required"][field_name][0])


def pick(pool: list[str], needles, what: str) -> str:
    for needle in needles:
        for f in pool:
            if needle.lower() in f.lower():
                return f
    raise ModelMissing("no %s found in ComfyUI (looked for %s)" % (what, ", ".join(needles)))


def resolve_backend_models(backend: ImageBackend, pools: dict, turbo: bool = True) -> dict:
    """Filenames for this backend, plus a `warnings` list for what was skipped."""
    out = {
        "unet": pick(pools.get("unet") or [], backend.unet, backend.label + " diffusion model"),
        "clip": pick(pools.get("clip") or [], backend.clip, backend.label + " text encoder"),
        "vae": pick(pools.get("vae") or [], backend.vae, backend.label + " VAE"),
        "lora": None,
        "warnings": [],
    }
    if turbo and backend.turbo_lora:
        try:
            out["lora"] = pick(pools.get("loras") or [], backend.turbo_lora, "turbo LoRA")
        except ModelMissing as exc:
            out["warnings"].append("%s - running at full step count" % exc)
    return out


# --- graph building ---------------------------------------------------------

@dataclass
class GraphOptions:
    prompt: str = ""
    negative: str = ""
    width: int = 1024
    height: int = 1024
    batch: int = 1
    steps: int = 20
    cfg: float = 2.5
    seed: int = 0
    denoise: float = 1.0
    turbo: bool = True
    edit_image: str | None = None    # filename relative to ComfyUI/input
    edit_megapixels: float = 1.0
    save_prefix: str = SAVE_PREFIX
    input_subdir: str = INPUT_SUBDIR
    pictures: list = field(default_factory=list)   # list[PackedPicture] | (slot, filename)


def _picture_file(p) -> str:
    return p.filename if hasattr(p, "filename") else p[1]


def build_graph(backend: ImageBackend, models: dict, opts: GraphOptions) -> dict:
    """API-format graph for `backend`. Node keys are readable: they become titles."""
    if opts.edit_image and not backend.supports_edit:
        raise ValueError("%s cannot edit an existing image" % backend.label)
    builder = {
        QWEN_EDIT.key: _build_qwen_edit,
        FLUX2_KLEIN.key: _build_flux2,
        QWEN_T2I.key: _build_qwen_t2i,
    }[backend.key]
    return builder(backend, models, opts)


def _loaders(backend: ImageBackend, models: dict, g: dict) -> str:
    """UNET (+ turbo LoRA, + sampling shift), CLIP and VAE. Returns the model key."""
    g["unet"] = {"class_type": "UNETLoader",
                 "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}}
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": models["clip"], "type": backend.clip_type,
                            "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}}

    model_key = "unet"
    if models.get("lora"):
        g["turbo_lora"] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"model": [model_key, 0],
                                      "lora_name": models["lora"], "strength_model": 1.0}}
        model_key = "turbo_lora"
    if backend.shift is not None:
        g["sampling_shift"] = {"class_type": "ModelSamplingAuraFlow",
                               "inputs": {"model": [model_key, 0], "shift": backend.shift}}
        model_key = "sampling_shift"
    return model_key


def _ref_loaders(g: dict, opts: GraphOptions, limit: int) -> list[str]:
    """A LoadImage per packed reference sheet, capped at `limit`. Returns node keys."""
    keys = []
    for p in opts.pictures[:max(0, limit)]:
        key = "ref_%d" % (len(keys) + 1)
        g[key] = {"class_type": "LoadImage",
                  "inputs": {"image": "%s/%s" % (opts.input_subdir, _picture_file(p))}}
        keys.append(key)
    return keys


def _edit_source(g: dict, opts: GraphOptions) -> str | None:
    """LoadImage -> megapixel clamp for the frame being edited."""
    if not opts.edit_image:
        return None
    g["edit_source"] = {"class_type": "LoadImage", "inputs": {"image": opts.edit_image}}
    g["edit_scaled"] = {"class_type": "ImageScaleToTotalPixels",
                        "inputs": {"image": ["edit_source", 0], "upscale_method": "lanczos",
                                   "megapixels": float(opts.edit_megapixels),
                                   "resolution_steps": 16}}
    return "edit_scaled"


def _tail(g: dict, latent_key: str, save_prefix: str) -> None:
    g["decode"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": [latent_key, 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": save_prefix}}


def _build_qwen_edit(backend, models, opts) -> dict:
    g: dict[str, dict] = {}
    model_key = _loaders(backend, models, g)
    g["cfg_norm"] = {"class_type": "CFGNorm",
                     "inputs": {"model": [model_key, 0], "strength": 1.0}}
    model_key = "cfg_norm"

    source = _edit_source(g, opts)
    # In edit mode the frame being edited owns slot 1, so only two reference
    # slots are left - which is also why the prompt writer is told to offset.
    refs = _ref_loaders(g, opts, backend.ref_slots - (1 if source else 0))
    slots = ([source] if source else []) + refs

    def encode(key: str, text: str) -> None:
        inputs = {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0]}
        for i, src in enumerate(slots[:3]):
            inputs["image%d" % (i + 1)] = [src, 0]
        g[key] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": inputs}

    encode("positive", opts.prompt)
    encode("negative", opts.negative)

    if source:
        g["latent"] = {"class_type": "VAEEncode",
                       "inputs": {"pixels": [source, 0], "vae": ["vae", 0]}}
    else:
        g["latent"] = {"class_type": "EmptySD3LatentImage",
                       "inputs": {"width": int(opts.width), "height": int(opts.height),
                                  "batch_size": int(opts.batch)}}

    g["sample"] = {"class_type": "KSampler",
                   "inputs": {"model": [model_key, 0], "seed": int(opts.seed),
                              "steps": int(opts.steps), "cfg": float(opts.cfg),
                              "sampler_name": backend.sampler, "scheduler": backend.scheduler,
                              "positive": ["positive", 0], "negative": ["negative", 0],
                              "latent_image": ["latent", 0], "denoise": float(opts.denoise)}}
    _tail(g, "sample", opts.save_prefix)
    return g


def _build_qwen_t2i(backend, models, opts) -> dict:
    g: dict[str, dict] = {}
    model_key = _loaders(backend, models, g)
    g["positive"] = {"class_type": "CLIPTextEncode",
                     "inputs": {"text": opts.prompt, "clip": ["clip", 0]}}
    g["negative"] = {"class_type": "CLIPTextEncode",
                     "inputs": {"text": opts.negative, "clip": ["clip", 0]}}
    g["latent"] = {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": int(opts.width), "height": int(opts.height),
                              "batch_size": int(opts.batch)}}
    g["sample"] = {"class_type": "KSampler",
                   "inputs": {"model": [model_key, 0], "seed": int(opts.seed),
                              "steps": int(opts.steps), "cfg": float(opts.cfg),
                              "sampler_name": backend.sampler, "scheduler": backend.scheduler,
                              "positive": ["positive", 0], "negative": ["negative", 0],
                              "latent_image": ["latent", 0], "denoise": float(opts.denoise)}}
    _tail(g, "sample", opts.save_prefix)
    return g


def _build_flux2(backend, models, opts) -> dict:
    """Flux.2 samples through SamplerCustomAdvanced + Flux2Scheduler, and takes
    its references as a chain of ReferenceLatent nodes on the positive branch."""
    g: dict[str, dict] = {}
    model_key = _loaders(backend, models, g)

    g["positive"] = {"class_type": "CLIPTextEncode",
                     "inputs": {"text": opts.prompt, "clip": ["clip", 0]}}
    cond_key = "positive"

    source = _edit_source(g, opts)
    if source:
        g["edit_latent"] = {"class_type": "VAEEncode",
                            "inputs": {"pixels": [source, 0], "vae": ["vae", 0]}}
        g["ref_edit"] = {"class_type": "ReferenceLatent",
                         "inputs": {"conditioning": [cond_key, 0], "latent": ["edit_latent", 0]}}
        cond_key = "ref_edit"

    for i, key in enumerate(_ref_loaders(g, opts, backend.ref_slots), start=1):
        enc, ref = "ref_latent_%d" % i, "ref_cond_%d" % i
        g[enc] = {"class_type": "VAEEncode",
                  "inputs": {"pixels": [key, 0], "vae": ["vae", 0]}}
        g[ref] = {"class_type": "ReferenceLatent",
                  "inputs": {"conditioning": [cond_key, 0], "latent": [enc, 0]}}
        cond_key = ref

    g["guidance"] = {"class_type": "FluxGuidance",
                     "inputs": {"conditioning": [cond_key, 0],
                                "guidance": float(backend.guidance or 4.0)}}

    # The edited frame is already encoded for its ReferenceLatent; sampling from
    # that same latent is what makes the output inherit the source's dimensions.
    latent_ref = ["edit_latent", 0] if source else ["latent", 0]
    if not source:
        g["latent"] = {"class_type": "EmptyFlux2LatentImage",
                       "inputs": {"width": int(opts.width), "height": int(opts.height),
                                  "batch_size": int(opts.batch)}}

    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(opts.seed)}}
    g["guider"] = {"class_type": "BasicGuider",
                   "inputs": {"model": [model_key, 0], "conditioning": ["guidance", 0]}}
    g["sampler"] = {"class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": backend.sampler}}
    g["sigmas"] = {"class_type": "Flux2Scheduler",
                   "inputs": {"steps": int(opts.steps), "width": int(opts.width),
                              "height": int(opts.height)}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                              "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                              "latent_image": latent_ref}}
    _tail(g, "sample", opts.save_prefix)
    return g


def solve_image_resolution(megapixels: float, aspect: str,
                           backend: ImageBackend | None = None) -> tuple[int, int]:
    return solve_resolution(megapixels, aspect,
                            multiple=(backend.multiple if backend else 16))
