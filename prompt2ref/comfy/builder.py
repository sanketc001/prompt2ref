"""Manifest + prompt -> a ready-to-open MiniMax H3 Ref2VA workflow."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .imagepack import PackedPicture, ScalePolicy, pack_entities
from .promptgen import build_prompt
from .template import DEFAULT_TEMPLATE, Ref2VATemplate, solve_resolution

log = logging.getLogger("prompt2ref.comfy.builder")

# Where ComfyUI reads LoadImage files from; overridable via config / --comfy-root.
DEFAULT_COMFY_ROOT = Path(r"C:\Users\micke\Desktop\AI\ComfyUI")
INPUT_SUBDIR = "prompt2ref"


@dataclass
class BuildOptions:
    # picture handling
    pack: str = "per-entity"
    scale: ScalePolicy = field(default_factory=ScalePolicy)
    labels: bool | None = None
    scale_per_tile: bool = False

    # video settings
    seconds: float = 1.0
    megapixels: float = 0.4
    aspect: str = "16:9"
    frames: int | None = None        # advanced: pin the frame count directly
    half_res: bool = False           # the template ships with this on; off = exact res

    # plumbing
    template: Path = DEFAULT_TEMPLATE
    comfy_root: Path | None = None
    out_path: Path | None = None
    install_to_comfy: bool = True    # also drop the workflow in ComfyUI's list
    keep_av_refs: bool = False       # leave <Video N> / <Audio N> groups untouched
    use_llm: bool = True


@dataclass
class BuildResult:
    workflow_path: Path
    installed_path: Path | None
    pictures: list[PackedPicture]
    prompt: str
    width: int
    height: int
    frames: int
    seconds: float
    fps: int
    warnings: list[str]

    def summary(self) -> str:
        lines = [
            "workflow : %s" % self.workflow_path,
            "video    : %dx%d  (%.2f MP)  %d frames  ~%.2fs @ %dfps"
            % (self.width, self.height, self.width * self.height / 1e6,
               self.frames, self.frames / self.fps, self.fps),
            "pictures : %d slot%s" % (len(self.pictures),
                                      "" if len(self.pictures) == 1 else "s"),
        ]
        for p in self.pictures:
            lines.append("   <Picture %d>  %-40s %dx%d  %s"
                         % (p.slot, p.filename, p.width, p.height, ", ".join(p.entities)))
        if self.installed_path:
            lines.append("installed: %s" % self.installed_path)
        for w in self.warnings:
            lines.append("warning  : %s" % w)
        return "\n".join(lines)


def _resolve_comfy_root(opts: BuildOptions) -> Path | None:
    root = opts.comfy_root or DEFAULT_COMFY_ROOT
    root = Path(root)
    return root if (root / "input").is_dir() else None


def build_workflow(
    manifest: dict,
    settings,
    opts: BuildOptions | None = None,
) -> BuildResult:
    """Render pictures, write the prompt, set video settings, save the workflow."""
    opts = opts or BuildOptions()
    warnings: list[str] = []

    comfy_root = _resolve_comfy_root(opts)
    if comfy_root is None:
        # Still produce everything, just staged locally, and say so.
        comfy_root = Path(settings.refs_dir).parent / "comfy_staging"
        (comfy_root / "input").mkdir(parents=True, exist_ok=True)
        warnings.append(
            "ComfyUI input folder not found - images staged in %s/input; copy them "
            "into your ComfyUI input directory" % comfy_root)

    tpl = Ref2VATemplate(opts.template)
    slots = len(tpl.pictures)

    # 1. pictures
    input_dir = comfy_root / "input" / INPUT_SUBDIR
    _clean_stale(input_dir)
    pictures, pack_warnings = pack_entities(
        manifest.get("entities", []),
        out_dir=input_dir,
        slots=slots,
        mode=opts.pack,
        policy=opts.scale,
        labels=opts.labels,
        scale_per_tile=opts.scale_per_tile,
    )
    warnings.extend(pack_warnings)

    tpl.clear_all_pictures()
    for p in pictures:
        tpl.set_picture(p.slot, "%s/%s" % (INPUT_SUBDIR, p.filename))
    if not opts.keep_av_refs:
        tpl.disable_av_references()

    # 2. prompt
    prompt_text = build_prompt(
        manifest.get("prompt", ""), pictures, opts.seconds, settings, use_llm=opts.use_llm)
    tpl.set_prompt(prompt_text)

    # 3. video settings
    tpl.set_half_res(opts.half_res)
    width, height = solve_resolution(opts.megapixels, opts.aspect)
    factor = tpl.half_res_factor
    if factor != 1.0:
        # ResolutionMaster feeds a downscale, so aim high enough to land on target
        tpl.set_resolution(int(round(width / factor)), int(round(height / factor)))
        warnings.append("half-resolution group is active - ResolutionMaster set to %dx%d "
                        "so the render lands at %dx%d"
                        % (round(width / factor), round(height / factor), width, height))
    else:
        tpl.set_resolution(width, height)

    tpl.set_duration_seconds(opts.seconds)
    frames = tpl.frames_for(opts.seconds)
    if opts.frames:
        if (opts.frames - 5) % 17:
            warnings.append("frame count %d is not congruent to 5 mod 17; the sampler "
                            "may reject it" % opts.frames)
        tpl.set_frames(opts.frames)
        frames = opts.frames
    elif abs(frames / 24 - opts.seconds) > 0.05:
        warnings.append("the template quantises length to 17k+5 frames: %.2fs -> %d frames "
                        "(~%.2fs at 24fps)" % (opts.seconds, frames, frames / 24))

    # 4. save
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(opts.out_path) if opts.out_path else (
        Path(settings.refs_dir).parent / "workflows" / ("ref2va-%s.json" % stamp))
    tpl.save(out_path)

    installed = None
    if opts.install_to_comfy:
        wf_dir = comfy_root / "user" / "default" / "workflows"
        if wf_dir.is_dir():
            installed = wf_dir / out_path.name
            shutil.copy2(out_path, installed)

    return BuildResult(
        workflow_path=out_path, installed_path=installed, pictures=pictures,
        prompt=prompt_text, width=width, height=height, frames=frames,
        seconds=opts.seconds, fps=24, warnings=warnings,
    )


def _clean_stale(input_dir: Path) -> None:
    """Drop previously generated slot files so an old <Picture 7> can't linger."""
    if not input_dir.is_dir():
        return
    for f in input_dir.glob("p2r_*.png"):
        f.unlink(missing_ok=True)


def load_manifest(path: str | Path | None, settings) -> dict:
    """Explicit path, else the newest run, else refs/latest.json."""
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    runs = sorted(Path(settings.runs_dir).glob("run-*.json"))
    if runs:
        return json.loads(runs[-1].read_text(encoding="utf-8"))
    latest = Path(settings.refs_dir) / "latest.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    raise FileNotFoundError("no manifest found - run prompt2ref on a prompt first")
