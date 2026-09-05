"""Actually render: submit a MiniMax H3 job to a running ComfyUI and wait.

The UI template can't be POSTed - `/prompt` only accepts the flat API graph, and
this template's SetNode/GetNode + subgraphs + rgthree bypassers don't convert
reliably. So the runner rebuilds the *executed* path as a minimal API graph
using the same reference images, prompt and video settings the builder produced,
mirroring the template's sampler (res_multistep / simple / 20 steps).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("prompt2ref.comfy.run")

DEFAULT_SERVER = "http://127.0.0.1:8188"

# Model files the MiniMax H3 path needs, matched against what the server reports.
WANT = {
    "unet": ["minimax_h3_ref2va", "minimax_h3"],
    "clip": ["qwen3vl_32b_minimax_h3"],
    "video_vae": ["minimax_h3_video_vae_fp16", "minimax_h3_video_vae"],
    "audio_vae": ["minimax_h3_audio_vae"],
}


class ComfyError(RuntimeError):
    pass


@dataclass
class RunResult:
    prompt_id: str
    outputs: list[Path]
    elapsed_s: float
    history: dict


def _get(server: str, path: str, timeout: float = 30):
    with urllib.request.urlopen(server + path, timeout=timeout) as r:
        return json.load(r)


def _post(server: str, path: str, payload: dict, timeout: float = 60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        server + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        raise ComfyError("%s -> %s: %s" % (path, exc.code, exc.read().decode()[:2000]))


def server_alive(server: str = DEFAULT_SERVER) -> dict | None:
    try:
        return _get(server, "/system_stats", timeout=6)
    except Exception:
        return None


def resolve_models(server: str = DEFAULT_SERVER) -> dict[str, str]:
    """Match the required models against the server's actual file lists."""
    unets = _get(server, "/object_info/UNETLoader")["UNETLoader"]["input"]["required"]["unet_name"][0]
    clips = _get(server, "/object_info/CLIPLoader")["CLIPLoader"]["input"]["required"]["clip_name"][0]
    vaes = _get(server, "/object_info/VAELoader")["VAELoader"]["input"]["required"]["vae_name"][0]

    def pick(pool, needles, what):
        for needle in needles:
            for f in pool:
                if needle in f.lower():
                    return f
        raise ComfyError("no %s model found on the server (looked for %s)" % (what, needles))

    return {
        "unet": pick(unets, WANT["unet"], "MiniMax H3 diffusion"),
        "clip": pick(clips, WANT["clip"], "MiniMax H3 text encoder"),
        "video_vae": pick(vaes, WANT["video_vae"], "MiniMax H3 video VAE"),
        "audio_vae": pick(vaes, WANT["audio_vae"], "MiniMax H3 audio VAE"),
    }


@dataclass
class PreviewOptions:
    """Live sampling preview via KJNodes' ModelPreviewOverrideKJ.

    The node decodes each sampler step with a tiny VAE and pushes it over
    ComfyUI's websocket as a `kj_preview_override` JSON message (base64 image),
    so a client sees the video forming instead of a progress bar.
    """

    enabled: bool = True
    tiny_vae: str = "taeh3.safetensors"   # "none" falls back to Latent2RGB
    max_resolution: int = 512
    jpeg_quality: int = 80
    frames: int = 4        # >1 gives an animated preview of the clip
    fps: int = 12

    @staticmethod
    def available(server: str = DEFAULT_SERVER) -> bool:
        try:
            return bool(_get(server, "/object_info/ModelPreviewOverrideKJ", timeout=15))
        except Exception:
            return False


def build_api_graph(
    pictures: list,          # list[PackedPicture] - or (slot, "subdir/file.png") pairs
    prompt: str,
    width: int,
    height: int,
    length: int,
    models: dict[str, str],
    seed: int = 0,
    steps: int = 20,
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    ref_image_size: str = "match",
    fps: float = 24.0,
    filename_prefix: str = "prompt2ref/ref2va",
    input_subdir: str = "prompt2ref",
    preview: PreviewOptions | None = None,
) -> dict:
    g: dict[str, dict] = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": models["clip"], "type": "minimax",
                            "device": "default"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": models["video_vae"]}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": models["audio_vae"]}},
    }

    mm_inputs = {
        "clip": ["clip", 0],
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "ref_image_size": ref_image_size,
        "vae": ["vvae", 0],
        "audio_vae": ["avae", 0],
    }
    for i, p in enumerate(pictures):
        name = p.filename if hasattr(p, "filename") else p[1]
        node = "img%d" % i
        g[node] = {"class_type": "LoadImage",
                   "inputs": {"image": "%s/%s" % (input_subdir, name)}}
        # Autogrow inputs are addressed by their flattened id (see comfy_api _io.py)
        mm_inputs["ref_images.ref_image_%d" % i] = [node, 0]

    g["mm"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": mm_inputs}

    model_ref = ["unet", 0]
    if preview and preview.enabled:
        g["preview"] = {
            "class_type": "ModelPreviewOverrideKJ",
            "inputs": {
                "model": ["unet", 0],
                "max_resolution": int(preview.max_resolution),
                "jpeg_quality": int(preview.jpeg_quality),
                "suppress_default_preview": True,
                "preview_frames": int(preview.frames),
                "preview_fps": int(preview.fps),
                "tiny_vae": preview.tiny_vae,
            },
        }
        model_ref = ["preview", 0]

    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    g["sigmas"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": model_ref, "scheduler": scheduler,
                              "steps": int(steps), "denoise": 1.0}}
    g["guider"] = {"class_type": "BasicGuider",
                   "inputs": {"model": model_ref, "conditioning": ["mm", 0]}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                              "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                              "latent_image": ["mm", 1]}}
    g["decode"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vvae", 0]}}
    g["decode_audio"] = {"class_type": "VAEDecodeAudio",
                         "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo",
                  "inputs": {"images": ["decode", 0], "fps": float(fps),
                             "audio": ["decode_audio", 0]}}
    g["save"] = {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": filename_prefix,
                            "format": "auto", "codec": "auto"}}
    return g


def submit(graph: dict, server: str = DEFAULT_SERVER, client_id: str | None = None) -> str:
    payload = {"prompt": graph, "client_id": client_id or uuid.uuid4().hex}
    res = _post(server, "/prompt", payload)
    if "prompt_id" not in res:
        raise ComfyError("unexpected /prompt response: %s" % json.dumps(res)[:800])
    return res["prompt_id"]


def wait(
    prompt_id: str,
    server: str = DEFAULT_SERVER,
    timeout_s: float = 7200,
    poll_s: float = 5.0,
    on_tick=None,
) -> RunResult:
    started = time.time()
    while True:
        elapsed = time.time() - started
        if elapsed > timeout_s:
            raise ComfyError("timed out after %.0fs waiting for %s" % (elapsed, prompt_id))
        try:
            hist = _get(server, "/history/" + prompt_id, timeout=20)
        except Exception as exc:
            log.debug("history poll failed: %s", exc)
            hist = {}
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error" or not status.get("completed", True):
                if status.get("status_str") == "error":
                    raise ComfyError(_error_text(status))
            if status.get("completed"):
                return RunResult(prompt_id, _outputs(entry, server), elapsed, entry)
        if on_tick:
            try:
                on_tick(_progress(server), elapsed)
            except Exception:
                pass
        time.sleep(poll_s)


def _progress(server: str) -> dict:
    try:
        q = _get(server, "/prompt", timeout=10)
        return {"queue_remaining": q.get("exec_info", {}).get("queue_remaining")}
    except Exception:
        return {}


def _error_text(status: dict) -> str:
    bits = []
    for kind, payload in status.get("messages", []):
        if kind in ("execution_error", "execution_interrupted"):
            bits.append("%s: %s %s" % (kind, payload.get("node_type", ""),
                                       payload.get("exception_message", "")))
            tb = payload.get("traceback")
            if tb:
                bits.append("".join(tb[-6:]))
    return "\n".join(bits) or json.dumps(status)[:1000]


def _outputs(entry: dict, server: str, comfy_root: Path | None = None) -> list[Path]:
    from .builder import DEFAULT_COMFY_ROOT

    root = Path(comfy_root) if comfy_root else DEFAULT_COMFY_ROOT
    files: list[Path] = []
    for out in entry.get("outputs", {}).values():
        for key in ("images", "gifs", "videos", "audio"):
            for item in out.get(key, []) or []:
                sub = item.get("subfolder", "")
                folder = item.get("type", "output")
                files.append(root / folder / sub / item["filename"])
    return files


def render(
    build_result,
    server: str = DEFAULT_SERVER,
    seed: int = 0,
    steps: int = 20,
    ref_image_size: str = "match",
    timeout_s: float = 7200,
    on_tick=None,
) -> RunResult:
    """One call: check the server, resolve models, submit the build, wait."""
    if server_alive(server) is None:
        raise ComfyError("no ComfyUI at %s - start it first" % server)
    models = resolve_models(server)
    graph = build_api_graph(
        build_result.pictures, build_result.prompt,
        build_result.width, build_result.height, build_result.frames,
        models, seed=seed, steps=steps, ref_image_size=ref_image_size,
        fps=float(build_result.fps), preview=None,
    )
    pid = submit(graph, server)
    return wait(pid, server, timeout_s=timeout_s, on_tick=on_tick)


# --- streaming render (live preview) ---------------------------------------

def _ws_url(server: str, client_id: str) -> str:
    base = server.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    return "%s/ws?clientId=%s" % (base, client_id)


def stream_render(
    build_result,
    server: str = DEFAULT_SERVER,
    seed: int = 0,
    steps: int = 20,
    ref_image_size: str = "match",
    preview: PreviewOptions | None = None,
    timeout_s: float = 7200,
    on_event=None,
    comfy_root: Path | None = None,
) -> RunResult:
    """Submit and relay ComfyUI's websocket events as they arrive.

    Emitted event types: `submitted`, `progress`, `preview`, `node`, `error`,
    `done`. The socket is opened *before* submitting so the first sampler step
    can't be missed.
    """
    from websockets.sync.client import connect

    def emit(payload):
        if on_event:
            try:
                on_event(payload)
            except Exception:
                log.debug("event handler raised", exc_info=True)

    if server_alive(server) is None:
        raise ComfyError("no ComfyUI at %s - start it first" % server)
    models = resolve_models(server)
    preview = preview or PreviewOptions()
    if preview.enabled and not PreviewOptions.available(server):
        preview.enabled = False
        emit({"type": "warning",
              "message": "ModelPreviewOverrideKJ not installed - no live preview"})

    graph = build_api_graph(
        build_result.pictures, build_result.prompt,
        build_result.width, build_result.height, build_result.frames,
        models, seed=seed, steps=steps, ref_image_size=ref_image_size,
        fps=float(build_result.fps), preview=preview,
    )

    client_id = uuid.uuid4().hex
    started = time.time()
    prompt_id = None
    with connect(_ws_url(server, client_id), max_size=64 * 1024 * 1024,
                 open_timeout=15) as ws:
        prompt_id = submit(graph, server, client_id=client_id)
        emit({"type": "submitted", "prompt_id": prompt_id,
              "steps": steps, "preview": preview.enabled})

        while True:
            elapsed = time.time() - started
            if elapsed > timeout_s:
                interrupt(server)
                raise ComfyError("timed out after %.0fs" % elapsed)
            try:
                raw = ws.recv(timeout=5)
            except TimeoutError:
                emit({"type": "waiting", "elapsed": round(elapsed, 1)})
                continue
            if isinstance(raw, (bytes, bytearray)):
                continue  # binary previews are suppressed by the KJ override
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue

            mtype, data = msg.get("type"), msg.get("data") or {}
            if data.get("prompt_id") not in (None, prompt_id):
                continue

            if mtype == "kj_preview_override":
                emit({
                    "type": "preview",
                    "image": data.get("image"),
                    "mime": data.get("mime", "image/jpeg"),
                    "step": data.get("step"), "total": data.get("total"),
                    "width": data.get("w"), "height": data.get("h"),
                    "step_ms": data.get("step_ms"), "avg_step_ms": data.get("avg_step_ms"),
                    "elapsed": round(elapsed, 1),
                })
            elif mtype in ("progress", "progress_state"):
                emit({"type": "progress", "value": data.get("value"),
                      "max": data.get("max"), "node": data.get("node"),
                      "elapsed": round(elapsed, 1)})
            elif mtype == "executing":
                node = data.get("node")
                emit({"type": "node", "node": node, "elapsed": round(elapsed, 1)})
                if node is None and data.get("prompt_id") == prompt_id:
                    break
            elif mtype == "execution_error":
                raise ComfyError("%s in %s: %s" % (
                    data.get("exception_type", "error"), data.get("node_type", "?"),
                    data.get("exception_message", "")))
            elif mtype in ("execution_interrupted", "execution_cached"):
                if mtype == "execution_interrupted":
                    raise ComfyError("interrupted")
            elif mtype == "execution_success":
                break

    elapsed = time.time() - started
    hist = _get(server, "/history/" + prompt_id, timeout=30).get(prompt_id, {})
    outputs = _outputs(hist, server, comfy_root)
    emit({"type": "done", "elapsed": round(elapsed, 1),
          "outputs": [str(p) for p in outputs]})
    return RunResult(prompt_id, outputs, elapsed, hist)


def interrupt(server: str = DEFAULT_SERVER) -> None:
    try:
        _post(server, "/interrupt", {}, timeout=10)
    except Exception as exc:
        log.debug("interrupt failed: %s", exc)
