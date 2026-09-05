"""Command line interface.

    python -m prompt2ref "Tom Cruise standing in front of the Taj Mahal holding an iPhone 16"
    python -m prompt2ref --dry-run "..."      # show entities only
    python -m prompt2ref --serve              # launch the web UI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .comfy import PACK_MODES
from .config import load_settings
from .entities import Entity
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prompt2ref",
        description="Turn a generation prompt into a local library of reference images.",
    )
    p.add_argument("prompt", nargs="*", help="the prompt text (omit with --serve or --stdin)")
    p.add_argument("--stdin", action="store_true", help="read the prompt from stdin")
    p.add_argument("--out", dest="refs_dir", help="output root (default ./refs)")
    p.add_argument("--per-shot", type=int, help="force N images per shot type")
    p.add_argument("--min-width", type=int, help="minimum accepted image width")
    p.add_argument("--min-height", type=int, help="minimum accepted image height")
    p.add_argument("--sources", help="comma-separated backend order, e.g. ddg,wikimedia")
    p.add_argument("--workers", type=int, help="entities fetched in parallel")
    p.add_argument("--model", help="Claude model for entity extraction")
    p.add_argument("--no-llm", action="store_true", help="skip Claude, use heuristic NER")
    p.add_argument("--force", action="store_true", help="refetch even if the folder is filled")
    p.add_argument("--verify", action="store_true",
                   help="Claude looks at each new image and drops watermarked / wrong-subject ones")
    p.add_argument("--dry-run", action="store_true", help="extract entities, download nothing")
    p.add_argument("--entities", help="JSON list of entities, bypassing extraction")
    p.add_argument("--json", action="store_true", help="print the manifest as JSON")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--serve", action="store_true", help="run the web UI instead")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8077)

    c = p.add_argument_group("ComfyUI - MiniMax H3 Ref2VA workflow")
    c.add_argument("--comfy", action="store_true",
                   help="after fetching, build a MiniMax H3 Ref2VA workflow")
    c.add_argument("--comfy-only", action="store_true",
                   help="build the workflow from an existing manifest, fetch nothing")
    c.add_argument("--manifest", help="manifest to build from (default: newest run)")
    c.add_argument("--pack", default="per-entity", choices=list(PACK_MODES),
                   help="how entities map onto the nine <Picture N> slots")
    c.add_argument("--ref-max", type=int, default=1024,
                   help="longest edge for reference images (default 1024; 0 = native)")
    c.add_argument("--ref-native", action="store_true",
                   help="pass reference images through at native resolution")
    c.add_argument("--ref-megapixels", type=float,
                   help="scale reference images to a megapixel budget instead")
    c.add_argument("--scale-per-tile", action="store_true",
                   help="apply the size cap to each tile instead of the finished sheet")
    c.add_argument("--labels", dest="labels", action="store_true", default=None,
                   help="draw entity names on multi-image sheets")
    c.add_argument("--no-labels", dest="labels", action="store_false",
                   help="never draw entity names")
    c.add_argument("--seconds", type=float, default=1.0, help="video duration (default 1)")
    c.add_argument("--frames", type=int, help="pin the frame count (must be 5 mod 17)")
    c.add_argument("--megapixels", type=float, default=0.4,
                   help="video resolution budget (default 0.4)")
    c.add_argument("--aspect", default="16:9", help="16:9, 9:16, 1:1, or w:h")
    c.add_argument("--half-res", action="store_true",
                   help="keep the template's half-resolution downscale enabled")
    c.add_argument("--keep-av", action="store_true",
                   help="leave the <Video N> / <Audio N> reference groups untouched")
    c.add_argument("--comfy-root", help="ComfyUI install dir (holds input/ and user/)")
    c.add_argument("--comfy-out", help="where to write the workflow JSON")
    c.add_argument("--no-install", action="store_true",
                   help="do not copy the workflow into ComfyUI's workflow list")
    c.add_argument("--template", help="a different Ref2VA template JSON to build from")
    c.add_argument("--render", action="store_true",
                   help="also submit the job to a running ComfyUI and wait for the video")
    c.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI address")
    c.add_argument("--steps", type=int, default=20, help="sampler steps (default 20)")
    c.add_argument("--seed", type=int, default=0, help="sampler seed")
    c.add_argument("--ref-size", default="match", choices=["match", "max"],
                   help="reference token budget; 'max' is sharper identity but far slower")
    c.add_argument("--live-preview", action="store_true",
                   help="stream per-step sampler previews (used by the web UI; "
                        "on the CLI it just gives step-accurate progress)")
    c.add_argument("--render-timeout", type=float, default=7200,
                   help="seconds to wait for the render (default 7200)")
    return p


# ASCII only - Windows consoles default to cp1252 and a raised UnicodeEncodeError
# inside a progress callback would silently swallow the line.
ICONS = {
    "extract_start": "...",
    "entities": "**",
    "search": "  ?",
    "image": "  +",
    "shot_start": " >",
    "shot_cached": " =",
    "shot_empty": " !",
    "entity_done": "OK",
    "entity_error": "XX",
    "rejected": " -",
    "refill": " ~",
    "done": "==",
}


def make_printer(quiet: bool):
    def emit(ev: dict):
        if quiet:
            return
        t = ev.get("type")
        icon = ICONS.get(t, " ")
        if t == "entities":
            names = ", ".join(
                "%s (%s)" % (e["name"], e["type"]) for e in ev["entities"]
            ) or "none found"
            print("%s %d entit%s: %s" % (icon, len(ev["entities"]),
                                         "y" if len(ev["entities"]) == 1 else "ies", names))
        elif t == "shot_start":
            print("%s %s / %s - fetching %d" % (icon, ev["entity"], ev["label"], ev["want"]))
        elif t == "shot_cached":
            print("%s %s / %s - already have %d" % (icon, ev["entity"], ev["shot"], ev["have"]))
        elif t == "search":
            print("%s %-50s %d hits" % (icon, ev["query"][:50], ev["hits"]))
        elif t == "image":
            print("%s %s/%s  %dx%d  [%s]" % (icon, ev["entity"], ev["file"],
                                             ev["width"], ev["height"], ev["source"]))
        elif t == "rejected":
            print("%s %s/%s rejected - %s" % (icon, ev["entity"], ev["file"], ev["reason"]))
        elif t == "refill":
            print("%s %s / %s - refilling %d" % (icon, ev["entity"], ev["shot"], ev["want"]))
        elif t == "shot_empty":
            print("%s %s / %s - nothing usable found" % (icon, ev["entity"], ev["shot"]))
        elif t == "entity_done":
            print("%s %s - %d images" % (icon, ev["entity"], ev["count"]))
        elif t == "entity_error":
            print("%s %s - %s" % (icon, ev["entity"], ev["error"]))
        elif t == "done":
            print("%s %d images total -> %s" % (icon, ev["total_images"], ev["manifest_path"]))
        sys.stdout.flush()

    return emit


def build_comfy(args, settings, manifest: dict) -> int:
    from .comfy import BuildOptions, ScalePolicy, build_workflow
    from .comfy.template import DEFAULT_TEMPLATE

    scale = ScalePolicy.native() if args.ref_native else ScalePolicy(
        max_edge=args.ref_max or None, megapixels=args.ref_megapixels)
    opts = BuildOptions(
        pack=args.pack, scale=scale, labels=args.labels,
        scale_per_tile=args.scale_per_tile,
        seconds=args.seconds, megapixels=args.megapixels, aspect=args.aspect,
        frames=args.frames, half_res=args.half_res,
        template=Path(args.template) if args.template else DEFAULT_TEMPLATE,
        comfy_root=Path(args.comfy_root) if args.comfy_root else settings.comfy_root,
        out_path=Path(args.comfy_out) if args.comfy_out else None,
        install_to_comfy=not args.no_install,
        keep_av_refs=args.keep_av,
        use_llm=not args.no_llm,
    )
    result = build_workflow(manifest, settings, opts)
    if not args.quiet:
        print()
        print(result.summary())
    if args.render:
        return render_comfy(args, result)
    if args.json:
        print(json.dumps({"workflow": str(result.workflow_path),
                          "installed": str(result.installed_path or ""),
                          "width": result.width, "height": result.height,
                          "frames": result.frames,
                          "pictures": [{"slot": p.slot, "file": p.filename,
                                        "entities": p.entities} for p in result.pictures],
                          "warnings": result.warnings}, indent=2))
    return 0


def render_comfy(args, result) -> int:
    """Submit the built job to ComfyUI and stream progress until the file exists."""
    import time

    from .comfy.run import ComfyError, PreviewOptions, stream_render

    print()
    print("submitting to %s ..." % args.server)
    started = time.time()
    state = {"step": 0, "total": args.steps}

    def on_event(ev):
        t = ev.get("type")
        if t == "submitted":
            print("  queued %s" % ev["prompt_id"][:8])
        elif t == "node":
            if ev.get("node"):
                print("  [%5.0fs] %s" % (ev["elapsed"], ev["node"]))
        elif t == "preview":
            state["step"] = ev.get("step") or state["step"]
            state["total"] = ev.get("total") or state["total"]
            avg = ev.get("avg_step_ms")
            print("  [%5.0fs] step %s/%s%s" % (
                ev["elapsed"], state["step"], state["total"],
                "  %.1fs/step" % (avg / 1000) if avg else ""))
        elif t == "progress" and ev.get("max"):
            print("  [%5.0fs] %s/%s" % (ev["elapsed"], ev["value"], ev["max"]))
        elif t == "warning":
            print("  ! %s" % ev["message"])
        sys.stdout.flush()

    try:
        run = stream_render(
            result, server=args.server, seed=args.seed, steps=args.steps,
            ref_image_size=args.ref_size, timeout_s=args.render_timeout,
            preview=PreviewOptions(enabled=args.live_preview),
            on_event=on_event)
    except ComfyError as exc:
        print("render failed after %.0fs:" % (time.time() - started))
        print(exc)
        return 1
    print()
    print("rendered in %.0fs" % run.elapsed_s)
    for f in run.outputs:
        print("  %s" % f)
    if not run.outputs:
        print("  (job completed but reported no output files)")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.serve:
        from .server import serve

        serve(host=args.host, port=args.port)
        return 0

    prompt = sys.stdin.read() if args.stdin else " ".join(args.prompt).strip()
    if not prompt and not args.comfy_only:
        build_parser().print_help()
        return 2

    settings = load_settings(
        refs_dir=args.refs_dir,
        per_shot=args.per_shot,
        min_width=args.min_width,
        min_height=args.min_height,
        workers=args.workers,
        model=args.model,
        verify=args.verify or None,
        sources=tuple(x.strip() for x in args.sources.split(",")) if args.sources else None,
    )
    pipe = Pipeline(settings)

    if args.comfy_only:
        from .comfy import load_manifest

        manifest = load_manifest(args.manifest, settings)
        if prompt:
            manifest["prompt"] = prompt
        return build_comfy(args, settings, manifest)

    entities = None
    if args.entities:
        entities = [Entity.from_dict(d) for d in json.loads(args.entities)]

    if args.dry_run:
        found = entities if entities is not None else pipe.extract(prompt, use_llm=not args.no_llm)
        payload = [e.to_dict() for e in found]
        print(json.dumps(payload, indent=2))
        return 0

    manifest = pipe.run(
        prompt,
        entities=entities,
        use_llm=not args.no_llm,
        force=args.force,
        on_event=make_printer(args.quiet or args.json),
    )
    if args.json and not args.comfy:
        print(json.dumps(manifest, indent=2))
    if args.comfy:
        return build_comfy(args, settings, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
