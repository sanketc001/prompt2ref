"""Write the saved demo workflows - one per thing the node pack can make.

Each demo is fully node-driven: the prompt goes in at the top, the P2R nodes
extract the entities, fetch and QC the references, pack them into the slots the
target model takes and write that model's prompt, and the bottom half is the
model itself. Nothing is pre-baked, so re-running a demo with a different prompt
re-collects different references.

    p2r_video_minimax_h3           prompt -> references -> H3 Ref2VA clip
    p2r_image_qwen_edit_generate   prompt -> references -> Qwen-Image-Edit still
    p2r_image_qwen_edit_edit       your frame + references -> Qwen-Image-Edit edit
    p2r_image_flux2_klein_generate prompt -> references -> Flux.2 Klein still

Run it with any Python that can import this repo (no torch needed):

    python comfyui_prompt2ref/build_demo_workflows.py [--server http://127.0.0.1:8188]

Graphs are written in API format and converted to the UI format by
prompt2ref.comfy.uigraph, which reads each node class's declared input order
from a running ComfyUI, from the node classes here, or from the bundled
schemas.json snapshot - in that order.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):          # importable from any working directory
    if p not in sys.path:
        sys.path.insert(0, p)

from prompt2ref.comfy.builder import DEFAULT_COMFY_ROOT  # noqa: E402
from prompt2ref.comfy.uigraph import save_ui_workflow  # noqa: E402

OUT_DIR = REPO / "workflows"

VIDEO_PROMPT = "Tom Cruise walks out of the Taj Mahal holding an iPhone 16 Pro, golden hour"
IMAGE_PROMPT = ("Zendaya on a Paris rooftop at blue hour, a Chanel No 5 bottle on the ledge "
                "beside her, the Eiffel Tower lit behind")
EDIT_PROMPT = "Replace the drink on the table with a bottle of Chanel No 5"

# Model files this machine has. They are matched by exact name because a demo
# workflow is opened, read and adjusted by hand - resolving them dynamically
# would just hide which checkpoint the graph was drawn against.
QWEN_EDIT_UNET = "qwen_image_edit_2511_nvfp4.safetensors"
QWEN_EDIT_LORA = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_nvfp4.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
FLUX2_UNET = "flux-2-klein-9b-nvfp4.safetensors"
FLUX2_CLIP = "qwen_3_8b_nvfp4.safetensors"   # Klein 9B wants Qwen3-8B, not the 4B
FLUX2_VAE = "flux2-vae.safetensors"
LOCAL_LLM_CLIP = "qwen3vl_8b_nvfp4.safetensors"


def p2r_front_half(prompt: str, entity_slots: int = 3, per_shot: int = 2) -> dict:
    """The half every demo shares: prompt -> verified references on disk.

    Both providers are in the graph. `provider_api` is the one wired up;
    `provider_local` sits beside it unwired, as the zero-cost swap - rewire its
    output into the three `provider` inputs to run the whole thing offline.
    """
    g = {
        "clip_local": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": LOCAL_LLM_CLIP, "type": "stable_diffusion"},
        },
        "provider_local": {
            "class_type": "P2RLocalProvider",
            "inputs": {"clip": ["clip_local", 0], "max_length": 768, "temperature": 0.4,
                       "top_p": 0.9, "top_k": 40, "seed": 0},
        },
        "provider_api": {
            "class_type": "P2RAnthropicProvider",
            "inputs": {"model": "claude-haiku-4-5", "max_tokens": 1024},
        },
        "extract": {
            "class_type": "P2RExtractEntities",
            "inputs": {"provider": ["provider_api", 0], "prompt": prompt, "use_llm": True},
        },
    }
    for i in range(1, 6):
        g["slot%d" % i] = {
            "class_type": "P2REntitySlot",
            "inputs": {"entities_json": ["extract", 0], "index": i - 1,
                       "enabled": i <= entity_slots},
        }
    g["combine"] = {
        "class_type": "P2RCombineEntities",
        "inputs": {"entity_json_%d" % i: ["slot%d" % i, 0] for i in range(1, 6)},
    }
    g["fetch"] = {
        "class_type": "P2RFetchReferences",
        "inputs": {"entities_json": ["combine", 0], "per_shot": per_shot,
                   "force": False, "sources": "ddg,wikimedia,openverse"},
    }
    g["verify"] = {
        "class_type": "P2RVerifyReferences",
        "inputs": {"provider": ["provider_api", 0], "manifest_json": ["fetch", 0],
                   "enabled": True},
    }
    return g


def image_front_half(prompt: str, slots: int, tagged: bool, mode: str,
                     editing_first: bool, aspect: str, megapixels: float = 1.0) -> dict:
    g = p2r_front_half(prompt)
    g["pack"] = {
        "class_type": "P2RPackReferencesImage",
        "inputs": {"manifest_json": ["verify", 0], "slots": slots,
                   "pack_mode": "primary", "overflow": "combine",
                   "ref_max": 1024, "labels": False},
    }
    g["image_prompt"] = {
        "class_type": "P2RBuildImagePrompt",
        "inputs": {"provider": ["provider_api", 0], "user_prompt": prompt,
                   "captions_json": ["pack", 4], "mode": mode,
                   "tagged_references": tagged, "editing_image_first": editing_first,
                   "style": "", "use_llm": True},
    }
    g["resolution"] = {
        "class_type": "P2RImageResolution",
        "inputs": {"megapixels": megapixels, "aspect": aspect, "multiple": 16},
    }
    return g


def qwen_edit_demo(edit: bool) -> dict:
    """Qwen-Image-Edit 2511: three conditioning slots, generation or editing.

    Editing spends the first slot on the frame being changed, so the references
    get the other two - and the prompt writer is told to start its tags at 2.
    """
    ref_slots = 2 if edit else 3
    g = image_front_half(
        EDIT_PROMPT if edit else IMAGE_PROMPT, slots=ref_slots, tagged=True,
        mode="edit" if edit else "generate", editing_first=edit,
        aspect="1:1" if edit else "16:9")

    g.update({
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": QWEN_EDIT_UNET, "weight_dtype": "default"}},
        # 4-step distillation. Bypass this node and put steps back to ~20 with
        # cfg 2.5 for the undistilled quality ceiling.
        "turbo_lora": {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": ["unet", 0], "lora_name": QWEN_EDIT_LORA,
                                  "strength_model": 1.0}},
        "sampling_shift": {"class_type": "ModelSamplingAuraFlow",
                           "inputs": {"model": ["turbo_lora", 0], "shift": 3.0}},
        "cfg_norm": {"class_type": "CFGNorm",
                     "inputs": {"model": ["sampling_shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
    })

    if edit:
        g["edit_source"] = {"class_type": "LoadImage", "inputs": {"image": "example.png"}}
        g["edit_scaled"] = {"class_type": "ImageScaleToTotalPixels",
                            "inputs": {"image": ["edit_source", 0], "upscale_method": "lanczos",
                                       "megapixels": 1.0, "resolution_steps": 16}}
        slot_sources = [["edit_scaled", 0], ["pack", 0], ["pack", 1]]
    else:
        slot_sources = [["pack", 0], ["pack", 1], ["pack", 2]]

    for key, text_slot in (("positive", 0), ("negative", 1)):
        inputs = {"clip": ["clip", 0], "prompt": ["image_prompt", text_slot],
                  "vae": ["vae", 0]}
        for i, src in enumerate(slot_sources, start=1):
            inputs["image%d" % i] = src
        g[key] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": inputs}

    if edit:
        # Sampling from the source's own latent is what keeps the output at the
        # source's size and framing.
        g["latent"] = {"class_type": "VAEEncode",
                       "inputs": {"pixels": ["edit_scaled", 0], "vae": ["vae", 0]}}
    else:
        g["latent"] = {"class_type": "EmptySD3LatentImage",
                       "inputs": {"width": ["resolution", 0], "height": ["resolution", 1],
                                  "batch_size": 1}}

    g["sample"] = {"class_type": "KSampler",
                   "inputs": {"model": ["cfg_norm", 0], "seed": 0, "steps": 4, "cfg": 1.0,
                              "sampler_name": "euler", "scheduler": "simple",
                              "positive": ["positive", 0], "negative": ["negative", 0],
                              "latent_image": ["latent", 0], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0],
                            "filename_prefix": "prompt2ref/qwen_edit"}}
    return g


def flux2_klein_demo() -> dict:
    """Flux.2 Klein: references become chained reference latents.

    Two reference slots rather than four, deliberately: every slot here is
    VAE-encoded unconditionally, so a slot the packer left empty would be a
    crash rather than a missing reference. `overflow: combine` guarantees both
    are filled for any prompt naming two or more entities.
    """
    g = image_front_half(IMAGE_PROMPT, slots=2, tagged=False, mode="generate",
                         editing_first=False, aspect="16:9")
    g.update({
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": FLUX2_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": FLUX2_CLIP, "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_VAE}},
        "positive": {"class_type": "CLIPTextEncode",
                     "inputs": {"text": ["image_prompt", 0], "clip": ["clip", 0]}},
    })

    cond = "positive"
    for i in range(2):
        enc, ref = "ref_latent_%d" % (i + 1), "ref_cond_%d" % (i + 1)
        g[enc] = {"class_type": "VAEEncode",
                  "inputs": {"pixels": ["pack", i], "vae": ["vae", 0]}}
        g[ref] = {"class_type": "ReferenceLatent",
                  "inputs": {"conditioning": [cond, 0], "latent": [enc, 0]}}
        cond = ref

    g.update({
        "guidance": {"class_type": "FluxGuidance",
                     "inputs": {"conditioning": [cond, 0], "guidance": 4.0}},
        "latent": {"class_type": "EmptyFlux2LatentImage",
                   "inputs": {"width": ["resolution", 0], "height": ["resolution", 1],
                              "batch_size": 1}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "guider": {"class_type": "BasicGuider",
                   "inputs": {"model": ["unet", 0], "conditioning": ["guidance", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "Flux2Scheduler",
                   "inputs": {"steps": 20, "width": ["resolution", 0],
                              "height": ["resolution", 1]}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                              "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                              "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0],
                            "filename_prefix": "prompt2ref/flux2_klein"}},
    })
    return g


def video_demo() -> dict:
    """The MiniMax H3 clip, as already validated end-to-end by test_workflow.py."""
    from test_workflow import graph

    return dict(graph)


# Node titles where a sentence reads better than the graph key. The edit demo
# ships pointed at ComfyUI's stock example.png purely so the node is not red on
# open - the title is what tells you to swap it.
TITLES = {
    "edit_source": "edit source - point this at your own frame",
    "provider_local": "provider: local CLIP (unwired - rewire to run offline)",
    "provider_api": "provider: Anthropic API (wired)",
    "extract": "extract entities from the prompt",
    "pack": "pack references into the model's slots",
    "image_prompt": "write the image / edit prompt",
}

DEMOS = {
    "p2r_video_minimax_h3": (video_demo, VIDEO_PROMPT,
                             "prompt -> references -> MiniMax H3 Ref2VA clip"),
    "p2r_image_qwen_edit_generate": (lambda: qwen_edit_demo(False), IMAGE_PROMPT,
                                     "prompt -> references -> Qwen-Image-Edit still"),
    "p2r_image_qwen_edit_edit": (lambda: qwen_edit_demo(True), EDIT_PROMPT,
                                 "your frame + references -> Qwen-Image-Edit edit; "
                                 "point 'edit source' at a photo with a drink on a table, "
                                 "or change the prompt to match your own frame"),
    "p2r_image_flux2_klein_generate": (flux2_klein_demo, IMAGE_PROMPT,
                                       "prompt -> references -> Flux.2 Klein still"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8188",
                    help="a running ComfyUI, for node schemas not in the snapshot")
    ap.add_argument("--out", default=str(OUT_DIR), help="where to write the workflows")
    ap.add_argument("--comfy-root", default=str(DEFAULT_COMFY_ROOT))
    ap.add_argument("--no-install", action="store_true",
                    help="do not copy them into ComfyUI's saved workflow list")
    ap.add_argument("--only", help="build just this demo")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    wf_dir = Path(args.comfy_root) / "user" / "default" / "workflows"

    for name, (build, prompt, blurb) in DEMOS.items():
        if args.only and args.only not in name:
            continue
        path = out_dir / (name + ".json")
        save_ui_workflow(build(), path, title=name, server=args.server, titles=TITLES,
                         extra={"prompt2ref_demo": blurb, "prompt2ref_prompt": prompt})
        line = "wrote %s" % path
        if not args.no_install and wf_dir.is_dir():
            shutil.copy2(path, wf_dir / path.name)
            line += "  ->  %s" % (wf_dir / path.name)
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
