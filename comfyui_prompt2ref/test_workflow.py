"""Build and submit the full local-node test workflow via ComfyUI's HTTP API.

Not a ComfyUI node - a standalone script used to validate the node package
end-to-end against the running ComfyUI server. Run with the comfy conda env:

    C:\\Users\\micke\\anaconda3\\envs\\comfy\\python.exe test_workflow.py
"""

import json
import sys
import urllib.request

SERVER = "http://127.0.0.1:8188"
TEST_PROMPT = "Tom Cruise walks out of the Taj Mahal holding an iPhone 16 Pro, golden hour"

graph = {
    "clip_local": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "qwen3vl_8b_nvfp4.safetensors", "type": "stable_diffusion"},
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
            "clip": ["clip_local", 0],
            "prompt": TEST_PROMPT,
            "use_llm": True,
            "max_length": 768,
            "temperature": 0.4,
            "seed": 0,
        },
    },
    "fetch": {
        "class_type": "P2RFetchReferences",
        "inputs": {
            "entities_json": ["extract", 0],
            "per_shot": 2,
            "force": False,
            "sources": "ddg,wikimedia,openverse",
        },
    },
    "verify": {
        "class_type": "P2RVerifyReferences",
        "inputs": {
            "clip": ["clip_local", 0],
            "manifest_json": ["fetch", 0],
            "enabled": True,
            "max_length": 512,
            "temperature": 0.2,
            "seed": 0,
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
            "clip": ["clip_local", 0],
            "user_prompt": TEST_PROMPT,
            "captions_json": ["pack", 9],
            "seconds": 1.0,
            "use_llm": True,
            "max_length": 1024,
            "temperature": 0.5,
            "seed": 0,
        },
    },
    "h3": {
        "class_type": "MiniMaxH3ReferenceToVideo",
        "inputs": {
            "clip": ["clip_h3", 0],
            "vae": ["vae_video", 0],
            "audio_vae": ["vae_audio", 0],
            "prompt": ["h3prompt", 0],
            "width": 320,
            "height": 192,
            "length": 5,
            "ref_image_size": "match",
            "ref_images": {
                "ref_image_0": ["pack", 0],
                "ref_image_1": ["pack", 1],
                "ref_image_2": ["pack", 2],
            },
        },
    },
    "decode": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["h3", 1], "vae": ["vae_video", 0]},
    },
    "save": {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": "p2r_node_test"},
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
