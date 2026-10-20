"""Manifest + prompt -> a ready-to-run reference-conditioned image workflow.

The same three steps as builder.py does for video - pack the references, write
the prompt, solve the size - but the output is an API graph built from scratch
for the chosen image backend (see imagegraph.py) plus the UI workflow rendered
from it, rather than a shipped template with its slots filled in.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .builder import INPUT_SUBDIR, _clean_stale, _resolve_comfy_root
from .imagegraph import (
    BACKENDS,
    DEFAULT_BACKEND,
    GraphOptions,
    ModelMissing,
    build_graph,
    list_models,
    resolve_backend_models,
    solve_image_resolution,
)
from .imageprompt import build_image_prompt
from .imagepack import PackedPicture, ScalePolicy, pack_entities
from .uigraph import save_ui_workflow

# Frames being edited live in their own folder: the packed-reference folder is
# wiped at the start of every build, and the thing you are editing must not be.
EDIT_SUBDIR = "prompt2ref_edit"


@dataclass
class ImageBuildOptions:
    backend: str = DEFAULT_BACKEND
    mode: str = "generate"           # generate | edit

    # reference handling - `primary` by default here, unlike video: an image
    # model conditioned on a contact sheet tends to draw the sheet.
    pack: str = "primary"
    scale: ScalePolicy = field(default_factory=ScalePolicy)
    labels: bool | None = None
    scale_per_tile: bool = False

    # image settings
    megapixels: float = 1.0
    aspect: str = "1:1"
    batch: int = 1
    steps: int | None = None         # None -> the backend's own default
    cfg: float | None = None
    seed: int = 0
    denoise: float = 1.0
    turbo: bool = True
    style: str = ""
    negative: str | None = None

    # editing
    edit_image: str | None = None    # a local path, or a name already in ComfyUI/input

    # plumbing
    comfy_root: Path | None = None
    out_path: Path | None = None
    install_to_comfy: bool = True
    server: str | None = None        # for /object_info model + schema lookups
    use_llm: bool = True


@dataclass
class ImageBuildResult:
    backend: str
    workflow_path: Path
    installed_path: Path | None
    graph: dict
    pictures: list[PackedPicture]
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    cfg: float
    seed: int
    batch: int
    models: dict
    edit_image: str | None
    warnings: list[str]

    def summary(self) -> str:
        b = BACKENDS[self.backend]
        lines = [
            "workflow : %s" % self.workflow_path,
            "backend  : %s" % b.label,
            "image    : %dx%d  (%.2f MP)  x%d  %d steps  cfg %.1f  seed %d"
            % (self.width, self.height, self.width * self.height / 1e6, self.batch,
               self.steps, self.cfg, self.seed),
            "models   : %s + %s + %s%s"
            % (self.models["unet"], self.models["clip"], self.models["vae"],
               " + " + self.models["lora"] if self.models.get("lora") else ""),
        ]
        if self.edit_image:
            lines.append("editing  : %s" % self.edit_image)
        lines.append("pictures : %d reference%s"
                     % (len(self.pictures), "" if len(self.pictures) == 1 else "s"))
        for p in self.pictures:
            lines.append("   ref %d  %-40s %dx%d  %s"
                         % (p.slot, p.filename, p.width, p.height, ", ".join(p.entities)))
        if self.installed_path:
            lines.append("installed: %s" % self.installed_path)
        for w in self.warnings:
            lines.append("warning  : %s" % w)
        return "\n".join(lines)


def build_image_workflow(
    manifest: dict,
    settings,
    opts: ImageBuildOptions | None = None,
) -> ImageBuildResult:
    opts = opts or ImageBuildOptions()
    warnings: list[str] = []

    backend = BACKENDS.get(opts.backend)
    if backend is None:
        raise ValueError("unknown image backend %r - have %s"
                         % (opts.backend, ", ".join(BACKENDS)))
    if opts.mode == "edit" and not opts.edit_image:
        raise ValueError("edit mode needs an image to edit")
    if opts.mode == "edit" and not backend.supports_edit:
        raise ValueError("%s cannot edit an existing image" % backend.label)

    comfy_root = _resolve_comfy_root(opts)
    if comfy_root is None:
        comfy_root = Path(settings.refs_dir).parent / "comfy_staging"
        (comfy_root / "input").mkdir(parents=True, exist_ok=True)
        warnings.append(
            "ComfyUI input folder not found - images staged in %s/input; copy them "
            "into your ComfyUI input directory" % comfy_root)

    # 1. references
    input_dir = comfy_root / "input" / INPUT_SUBDIR
    _clean_stale(input_dir)
    edit_name = _stage_edit_image(opts.edit_image, comfy_root) if opts.mode == "edit" else None

    slots = backend.ref_slots - (1 if edit_name else 0)
    pictures: list[PackedPicture] = []
    if slots > 0:
        pictures, pack_warnings = pack_entities(
            manifest.get("entities", []), out_dir=input_dir, slots=slots,
            mode=opts.pack, policy=opts.scale, labels=opts.labels,
            scale_per_tile=opts.scale_per_tile)
        warnings.extend(pack_warnings)
    elif manifest.get("entities"):
        warnings.append(
            "%s takes no reference images - the references only shape the written prompt"
            % backend.label)

    # 2. prompt
    prompt, negative = build_image_prompt(
        manifest.get("prompt", ""), pictures, settings,
        mode=opts.mode, tagged=backend.tagged_refs,
        edit_offset=1 if edit_name else 0,
        style=opts.style, negative=opts.negative, use_llm=opts.use_llm)

    # 3. models + size
    pools = list_models(server=opts.server, comfy_root=comfy_root)
    try:
        models = resolve_backend_models(backend, pools, turbo=opts.turbo)
    except ModelMissing as exc:
        raise ModelMissing("%s: %s" % (backend.label, exc))
    warnings.extend(models.pop("warnings", []))

    steps, cfg = backend.defaults(opts.turbo and bool(models.get("lora")))
    steps = opts.steps or steps
    cfg = opts.cfg if opts.cfg is not None else cfg
    width, height = solve_image_resolution(opts.megapixels, opts.aspect, backend)

    graph = build_graph(backend, models, GraphOptions(
        prompt=prompt, negative=negative, width=width, height=height,
        batch=opts.batch, steps=steps, cfg=cfg, seed=opts.seed,
        denoise=opts.denoise, turbo=opts.turbo, edit_image=edit_name,
        edit_megapixels=opts.megapixels, pictures=pictures,
    ))

    # 4. save
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(opts.out_path) if opts.out_path else (
        Path(settings.refs_dir).parent / "workflows"
        / ("p2r-image-%s-%s.json" % (backend.key, stamp)))
    try:
        save_ui_workflow(graph, out_path, title="prompt2ref-image-%s" % backend.key,
                         server=opts.server,
                         extra={"prompt2ref_prompt": manifest.get("prompt", ""),
                                "prompt2ref_mode": opts.mode})
    except KeyError as exc:
        warnings.append("UI workflow not written (%s) - the API graph still renders" % exc)

    installed = None
    if opts.install_to_comfy and out_path.exists():
        wf_dir = comfy_root / "user" / "default" / "workflows"
        if wf_dir.is_dir():
            installed = wf_dir / out_path.name
            shutil.copy2(out_path, installed)

    return ImageBuildResult(
        backend=backend.key, workflow_path=out_path, installed_path=installed,
        graph=graph, pictures=pictures, prompt=prompt, negative=negative,
        width=width, height=height, steps=steps, cfg=cfg, seed=opts.seed,
        batch=opts.batch, models=models, edit_image=edit_name, warnings=warnings,
    )


def edit_dir(comfy_root: Path) -> Path:
    return Path(comfy_root) / "input" / EDIT_SUBDIR


def _stage_edit_image(source: str | None, comfy_root: Path) -> str | None:
    """Put the frame being edited where ComfyUI's LoadImage can see it.

    A name that already resolves inside ComfyUI's input tree is used as-is;
    anything else is copied into the edit folder.
    """
    if not source:
        return None
    src = Path(source)
    comfy_input = Path(comfy_root) / "input"
    if not src.is_absolute() and (comfy_input / source).exists():
        return str(source).replace("\\", "/")
    if not src.exists():
        raise FileNotFoundError("no such image to edit: %s" % source)
    dest_dir = edit_dir(comfy_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("p2r_edit_%s%s" % (datetime.now().strftime("%H%M%S"),
                                          src.suffix.lower() or ".png"))
    shutil.copy2(src, dest)
    return "%s/%s" % (EDIT_SUBDIR, dest.name)
