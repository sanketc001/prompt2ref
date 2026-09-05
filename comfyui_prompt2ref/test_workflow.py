"""Build and submit the full local-node + real-video-render test workflow via
ComfyUI's HTTP API. Not a ComfyUI node - a standalone script used to validate
the node package end-to-end against the running ComfyUI server. Run with the
comfy conda env:

    C:\\Users\\micke\\anaconda3\\envs\\comfy\\python.exe test_workflow.py
"""

import json
import sys
import urllib.request

SERVER = "http://127.0.0.1:8188"
TEST_PROMPT = "Tom Cruise walks out of the Taj Mahal holding an iPhone 16 Pro, golden hour"

# Skips the optional LTX-2.3-22B refine pass the source template has after H3 -
# that's a separate quality upscale, not "the video generation" itself, and
# doubles the VRAM/model-load cost. Real H3 sampler -> decode -> video file only.
STEPS = 8            # turbo LoRA is a 4-8 step distillation
WIDTH = 1344         # ~1 megapixel at 16:9 (the model's own trained/default resolution)
HEIGHT = 768
LENGTH = 124          # 5 mod 17, ~5s at 24fps

graph = {
    # Two alternative providers for P2RExtractEntities/P2RVerifyReferences/
    # P2RBuildH3Prompt - "provider_api" is wired into the pipeline below;
    # "provider_local" is left in the graph, unwired, as the free/local
    # alternative to swap back in (rewire its output into the three
    # `provider` inputs currently pointed at provider_api).
    "clip_local": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "qwen3vl_8b_nvfp4.safetensors", "type": "stable_diffusion"},
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
    "clip_h3": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors", "type": "minimax"},
    },
    "vae_video": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
    },
    "vae_audio": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
    },
    "extract": {
        "class_type": "P2RExtractEntities",
        "inputs": {
            "provider": ["provider_api", 0],
            "prompt": TEST_PROMPT,
            "use_llm": True,
        },
    },
    # 5 switchable entity slots: slot N starts from the Nth auto-extracted
    # entity, can be renamed/retyped/redescribed, or switched off entirely
    # (a disabled/empty slot contributes nothing - no gap in <Picture N>).
    "slot1": {"class_type": "P2REntitySlot",
              "inputs": {"entities_json": ["extract", 0], "index": 0, "enabled": True}},
    "slot2": {"class_type": "P2REntitySlot",
              "inputs": {"entities_json": ["extract", 0], "index": 1, "enabled": True}},
    "slot3": {"class_type": "P2REntitySlot",
              "inputs": {"entities_json": ["extract", 0], "index": 2, "enabled": True}},
    "slot4": {"class_type": "P2REntitySlot",
              "inputs": {"entities_json": ["extract", 0], "index": 3, "enabled": False}},
    "slot5": {"class_type": "P2REntitySlot",
              "inputs": {"entities_json": ["extract", 0], "index": 4, "enabled": False}},
    "combine": {
        "class_type": "P2RCombineEntities",
        "inputs": {
            "entity_json_1": ["slot1", 0], "entity_json_2": ["slot2", 0],
            "entity_json_3": ["slot3", 0], "entity_json_4": ["slot4", 0],
            "entity_json_5": ["slot5", 0],
        },
    },
    "fetch": {
        "class_type": "P2RFetchReferences",
        "inputs": {
            "entities_json": ["combine", 0],
            "per_shot": 2,
            "force": False,
            "sources": "ddg,wikimedia,openverse",
        },
    },
    "verify": {
        "class_type": "P2RVerifyReferences",
        "inputs": {
            "provider": ["provider_api", 0],
            "manifest_json": ["fetch", 0],
            "enabled": True,
        },
    },
    "pack": {
        "class_type": "P2RPackReferences",
        "inputs": {
            "manifest_json": ["verify", 0],
            "pack_mode": "per-entity",
            "ref_max": 1024,
            "labels": False,
        },
    },
    "h3prompt": {
        "class_type": "P2RBuildH3Prompt",
        "inputs": {
            "provider": ["provider_api", 0],
            "user_prompt": TEST_PROMPT,
            "captions_json": ["pack", 9],
            "seconds": 1.0,
            "use_llm": True,
        },
    },
    "h3": {
        "class_type": "MiniMaxH3ReferenceToVideo",
        "inputs": {
            "clip": ["clip_h3", 0],
            "vae": ["vae_video", 0],
            "audio_vae": ["vae_audio", 0],
            "prompt": ["h3prompt", 0],
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "ref_image_size": "match",
            "ref_images": {
                "ref_image_0": ["pack", 0],
                "ref_image_1": ["pack", 1],
                "ref_image_2": ["pack", 2],
                "ref_image_3": ["pack", 3],
                "ref_image_4": ["pack", 4],
            },
        },
    },
    # --- real sampling -> decode -> video, replacing the earlier smoke-test
    # VAEDecode+SaveImage tail. Mirrors templates/minimax_h3_ref2va.json's
    # subgraph 904, flattened, skipping its optional LTX-refine second stage.
    "unet": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors", "weight_dtype": "default"},
    },
    "lora": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["unet", 0],
            "lora_name": "minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors",
            "strength_model": 1.0,
        },
    },
    "noise": {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": 0},
    },
    "scheduler": {
        "class_type": "BasicScheduler",
        "inputs": {"model": ["lora", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0},
    },
    "guider": {
        "class_type": "BasicGuider",
        "inputs": {"model": ["lora", 0], "conditioning": ["h3", 0]},
    },
    "sampler_select": {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "res_multistep"},
    },
    "sample": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler_select", 0],
            "sigmas": ["scheduler", 0],
            "latent_image": ["h3", 1],
        },
    },
    "decode_video": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sample", 1], "vae": ["vae_video", 0]},  # denoised_output
    },
    "decode_audio": {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["sample", 1], "vae": ["vae_audio", 0]},
    },
    "video": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["decode_video", 0],
            "audio": ["decode_audio", 0],
            "frame_rate": 24,
            "loop_count": 0,
            "filename_prefix": "p2r_node_test",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "save_metadata": True,
            "pingpong": False,
            "save_output": True,
        },
    },
}


def main():
    payload = json.dumps({"prompt": graph}).encode("utf-8")
    req = urllib.request.Request(SERVER + "/prompt", data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            print(resp.status, resp.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code)
        print(e.read().decode())
        sys.exit(1)


if __name__ == "__main__":
    main()
