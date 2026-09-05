"""Local web UI - the prompt interface.

    python -m prompt2ref --serve        ->  http://127.0.0.1:8077

Endpoints (all local, no auth - this is a workstation tool):
    GET  /                     the single-page UI
    POST /api/extract          prompt -> entities (editable before fetching)
    POST /api/fetch            entities -> images, streamed as NDJSON progress
    GET  /api/library          every entity folder currently on disk
    GET  /api/manifest         the most recent run manifest
    DELETE /api/image          drop one image and re-write entity.json
    GET  /files/<path>         serves the refs directory
"""

from __future__ import annotations

import dataclasses
import json
import queue
import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_settings
from .entities import Entity
from .pipeline import ENTITY_FILE, Pipeline

STATIC_DIR = Path(__file__).parent / "static"
NEWLINE = chr(10)


class ExtractRequest(BaseModel):
    prompt: str = ""
    use_llm: bool = True
    model: str | None = None


class FetchRequest(BaseModel):
    prompt: str = ""
    entities: list[dict] | None = None
    use_llm: bool = True
    force: bool = False
    per_shot: int | None = None
    verify: bool = False
    model: str | None = None


class DeleteRequest(BaseModel):
    slug: str
    file: str


class ComfyRequest(BaseModel):
    """Build a MiniMax H3 Ref2VA workflow from the latest (or given) manifest."""

    manifest: str | None = None
    prompt: str = ""
    pack: str = "per-entity"
    ref_max: int | None = 1024
    ref_megapixels: float | None = None
    ref_native: bool = False
    scale_per_tile: bool = False
    labels: bool | None = None
    seconds: float = 1.0
    megapixels: float = 0.4
    aspect: str = "16:9"
    frames: int | None = None
    half_res: bool = False
    keep_av: bool = False
    use_llm: bool = True
    model: str | None = None


class RenderRequest(ComfyRequest):
    """Build, then submit to ComfyUI and stream progress + live preview."""

    server: str = "http://127.0.0.1:8188"
    steps: int = 20
    seed: int = 0
    ref_image_size: str = "match"          # MiniMax token budget: match | max
    live_preview: bool = True
    preview_frames: int = 4
    preview_fps: int = 12
    preview_max: int = 512
    timeout_s: float = 7200


def create_app(settings=None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Prompt2Ref", version="0.1.0")
    pipe = Pipeline(settings)

    app.mount("/files", StaticFiles(directory=str(settings.refs_dir)), name="files")
    app.state.settings = settings
    app.state.pipeline = pipe

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    def config():
        return {
            "refs_dir": str(settings.refs_dir),
            "sources": list(settings.sources),
            "model": settings.model,
            "llm_available": bool(settings.api_key),
            "verify_default": settings.verify,
        }

    @app.get("/api/models")
    def models():
        """Anthropic models this API key can use, for the model-picker dropdown."""
        if not settings.api_key:
            return {"models": [], "error": "no ANTHROPIC_API_KEY configured"}
        import anthropic

        try:
            client = anthropic.Anthropic(api_key=settings.api_key)
            found = [{"id": m.id, "display_name": m.display_name}
                     for m in client.models.list(limit=1000)]
        except Exception as exc:
            raise HTTPException(502, "failed to list models: %s" % exc)
        found.sort(key=lambda m: m["display_name"])
        return {"models": found}

    def _with_model(req) -> object:
        """Per-request settings so one client's model choice can't leak into another's."""
        return dataclasses.replace(settings, model=req.model) if req.model else settings

    @app.post("/api/extract")
    def extract(req: ExtractRequest):
        if not req.prompt.strip():
            raise HTTPException(400, "prompt is empty")
        run_pipe = pipe if not req.model else Pipeline(_with_model(req))
        found = run_pipe.extract(req.prompt, use_llm=req.use_llm)
        return {"entities": [e.to_dict() for e in found]}

    @app.post("/api/fetch")
    def fetch(req: FetchRequest):
        entities = (
            [Entity.from_dict(d) for d in req.entities]
            if req.entities is not None
            else None
        )
        # per-request settings so one client's --verify/model choice can't leak into another's
        run_pipe = Pipeline(dataclasses.replace(_with_model(req), verify=req.verify))
        events: queue.Queue = queue.Queue()
        SENTINEL = object()

        def worker():
            try:
                manifest = run_pipe.run(
                    req.prompt,
                    entities=entities,
                    use_llm=req.use_llm,
                    force=req.force,
                    per_shot=req.per_shot,
                    on_event=events.put,
                )
                events.put({"type": "manifest", "manifest": manifest})
            except Exception as exc:  # surface it in the UI rather than hanging
                events.put({"type": "fatal", "error": str(exc)})
            finally:
                events.put(SENTINEL)

        threading.Thread(target=worker, daemon=True).start()

        def stream():
            while True:
                item = events.get()
                if item is SENTINEL:
                    break
                yield json.dumps(item, default=str) + "\n"

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _build(req: ComfyRequest):
        from .comfy import BuildOptions, ScalePolicy, build_workflow, load_manifest

        build_settings = _with_model(req)
        try:
            manifest = load_manifest(req.manifest, build_settings)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc))
        if req.prompt.strip():
            manifest["prompt"] = req.prompt

        scale = ScalePolicy.native() if req.ref_native else ScalePolicy(
            max_edge=req.ref_max or None, megapixels=req.ref_megapixels)
        opts = BuildOptions(
            pack=req.pack, scale=scale, labels=req.labels,
            scale_per_tile=req.scale_per_tile,
            seconds=req.seconds, megapixels=req.megapixels, aspect=req.aspect,
            frames=req.frames, half_res=req.half_res, keep_av_refs=req.keep_av,
            comfy_root=build_settings.comfy_root, use_llm=req.use_llm,
        )
        result = build_workflow(manifest, build_settings, opts)
        app.state.packed_dir = result.pictures[0].path.parent if result.pictures else None
        return result

    def _build_payload(result) -> dict:
        return {
            "workflow": str(result.workflow_path),
            "installed": str(result.installed_path) if result.installed_path else None,
            "width": result.width, "height": result.height,
            "frames": result.frames, "fps": result.fps,
            "seconds_effective": round(result.frames / result.fps, 2),
            "prompt": result.prompt,
            "warnings": result.warnings,
            "pictures": [
                {"slot": p.slot, "file": p.filename, "width": p.width, "height": p.height,
                 "entities": p.entities, "caption": p.caption}
                for p in result.pictures
            ],
        }

    @app.post("/api/comfy")
    def comfy(req: ComfyRequest):
        return _build_payload(_build(req))

    @app.post("/api/render")
    def render(req: RenderRequest):
        """Build the workflow, submit it, and stream progress + preview frames."""
        from .comfy.run import ComfyError, PreviewOptions, interrupt, stream_render

        result = _build(req)
        events: queue.Queue = queue.Queue()
        SENTINEL = object()
        app.state.render_server = req.server

        def worker():
            try:
                events.put({"type": "build", "build": _build_payload(result)})
                run = stream_render(
                    result, server=req.server, seed=req.seed, steps=req.steps,
                    ref_image_size=req.ref_image_size,
                    preview=PreviewOptions(
                        enabled=req.live_preview, frames=req.preview_frames,
                        fps=req.preview_fps, max_resolution=req.preview_max),
                    timeout_s=req.timeout_s, on_event=events.put,
                    comfy_root=settings.comfy_root,
                )
                events.put({
                    "type": "result",
                    "elapsed": run.elapsed_s,
                    "outputs": [
                        {"path": str(p), "name": p.name,
                         "url": "/api/output?path=" + quote(str(p))}
                        for p in run.outputs
                    ],
                })
            except ComfyError as exc:
                events.put({"type": "error", "error": str(exc)})
            except Exception as exc:
                events.put({"type": "error", "error": "%s: %s" % (type(exc).__name__, exc)})
            finally:
                events.put(SENTINEL)

        threading.Thread(target=worker, daemon=True).start()

        def stream():
            while True:
                item = events.get()
                if item is SENTINEL:
                    break
                yield json.dumps(item, default=str) + NEWLINE

        return StreamingResponse(
            stream(), media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/render/cancel")
    def render_cancel():
        from .comfy.run import interrupt

        interrupt(getattr(app.state, "render_server", "http://127.0.0.1:8188"))
        return {"ok": True}

    @app.get("/api/output")
    def output(path: str):
        """Serve a rendered file. Restricted to the ComfyUI output tree."""
        from .comfy.builder import DEFAULT_COMFY_ROOT

        root = (Path(settings.comfy_root) if settings.comfy_root else DEFAULT_COMFY_ROOT)
        target = Path(path).resolve()
        allowed = (root / "output").resolve()
        if not str(target).startswith(str(allowed)) or not target.exists():
            raise HTTPException(404, "not an output file")
        return FileResponse(target)

    @app.get("/api/comfy/preview/{name}")
    def comfy_preview(name: str):
        packed = getattr(app.state, "packed_dir", None)
        if not packed:
            raise HTTPException(404, "nothing built yet")
        target = (Path(packed) / name).resolve()
        if not str(target).startswith(str(Path(packed).resolve())) or not target.exists():
            raise HTTPException(404, "no such picture")
        return FileResponse(target)

    @app.get("/api/library")
    def library():
        return pipe.rebuild_index()

    @app.get("/api/entity/{slug}")
    def entity(slug: str):
        f = settings.entity_dir(slug) / ENTITY_FILE
        if not f.exists():
            raise HTTPException(404, "unknown entity " + slug)
        return JSONResponse(json.loads(f.read_text(encoding="utf-8")))

    @app.get("/api/manifest")
    def manifest():
        f = settings.refs_dir / "latest.json"
        if not f.exists():
            return {"entities": [], "prompt": ""}
        return JSONResponse(json.loads(f.read_text(encoding="utf-8")))

    @app.delete("/api/image")
    def delete_image(req: DeleteRequest):
        dest = settings.entity_dir(req.slug)
        target = (dest / req.file).resolve()
        if not str(target).startswith(str(dest.resolve())):
            raise HTTPException(400, "path escapes the entity folder")
        if target.exists():
            target.unlink()
        record_file = dest / ENTITY_FILE
        if record_file.exists():
            rec = json.loads(record_file.read_text(encoding="utf-8"))
            rec["images"] = [i for i in rec.get("images", []) if i.get("file") != req.file]
            if rec.get("primary", "").endswith("/" + req.file):
                rec["primary"] = rec["images"][0]["path"] if rec["images"] else None
            record_file.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        pipe.rebuild_index()
        return {"ok": True}

    return app


def serve(host: str = "127.0.0.1", port: int = 8077, settings=None) -> None:
    import uvicorn

    print("Prompt2Ref UI  ->  http://%s:%d" % (host, port))
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")
