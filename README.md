# Prompt2Ref

Type a generation prompt, get a local folder of reference images for every named
entity in it — then hand those references straight to a **video** model
(MiniMax H3 Ref2VA), an **image** model (Qwen-Image-Edit, Flux.2 Klein), or an
**edit** of a frame you already have.

```
"Cinematic shot of Sherlock Holmes in front of the Taj Mahal holding an iPhone 16 Pro"

refs/
  sherlock-holmes/     portrait_01.jpg  portrait_02.jpg  fullbody_01.jpg  fullbody_02.jpg  profile_01.jpg
  taj-mahal/      establishing_01.jpg  establishing_02.jpg  detail_01.jpg
  iphone-16-pro/  product_01.jpg  product_02.jpg  context_01.jpg
  index.json      latest.json
runs/
  run-20260905-024816.json      <- the manifest your workflow reads
```

Each entity folder also holds an `entity.json` with per-image provenance
(source URL, page, credit, license, the query that found it, sha256 + perceptual hash).

---

## What it does

1. **Extract** — Claude reads the prompt and returns typed, canonicalised entities.
   `"the Taj"` → `Taj Mahal (landmark)`, `"iPhone 16"` → `iPhone 16 (product)`.
   A keyless capitalisation heuristic is the fallback when there's no API key.
2. **Plan shots** — the entity *type* decides which angles are worth having:

   | type | shots collected |
   |---|---|
   | `person` | closeup portrait ×2, full body ×2, profile ×1 |
   | `fictional_character` | full body ×2, closeup ×2 |
   | `landmark` / `place` | wide establishing ×2, detail ×1 |
   | `product` | product shot ×2, in-use ×1 |
   | `organization` | logo ×2 (alpha preserved as PNG), context ×1 |
   | `vehicle` | exterior 3/4 ×2, interior ×1 |
   | `animal` | full body ×2, head closeup ×1 |
   | `artwork` | the work ×2 |
   | `food` | plated ×2, overhead ×1 |
   | `event` / `other` | generic reference ×3 |

3. **Search** — DuckDuckGo Images, Wikimedia Commons and Openverse by default
   (no keys). SerpAPI or Google CSE jump to the front of the queue automatically
   if their env keys are present.
4. **Download & filter** — resolution and aspect gates per shot, watermarked
   stock hosts blocked outright, near-duplicates rejected by perceptual hash,
   EXIF-rotated and re-encoded to clean JPEG (or PNG when transparency matters).
5. **Optionally verify** — `--verify` shows every new frame back to Claude and
   deletes anything watermarked, off-subject or of the wrong shot type, then
   refills the hole.
6. **Manifest** — one JSON per run listing every entity, every file and a
   `primary` pick, for the downstream workflow to consume.

---

## Setup

```bash
pip install -r requirements.txt
```

`.env` in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...

# optional — enables better image search, auto-detected
SERPAPI_API_KEY=...
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...

# optional defaults
PROMPT2REF_MODEL=claude-opus-5
PROMPT2REF_SOURCES=ddg,wikimedia,openverse
PROMPT2REF_VERIFY=1
PROMPT2REF_REFS_DIR=D:/refs
```

## Use it

**Web UI** — the prompt interface: paste a prompt, review and edit the detected
entities (name, type, disambiguator) before anything downloads, watch images
land live, delete bad ones, browse the accumulated library.

```bash
python -m prompt2ref --serve          # http://127.0.0.1:8077
```

**CLI**

```bash
python -m prompt2ref "Sherlock Holmes in front of the Taj Mahal holding an iPhone 16 Pro"
python -m prompt2ref --dry-run "..."          # entities only, nothing downloaded
python -m prompt2ref --verify --per-shot 3 "..."
python -m prompt2ref --force "..."            # ignore what's cached, refetch
python -m prompt2ref --json "..." > manifest.json
python -m prompt2ref --no-llm "..."           # heuristic NER, zero API calls
```

**Python**

```python
from prompt2ref import build_references

manifest = build_references(
    "Sherlock Holmes in front of the Taj Mahal holding an iPhone 16 Pro",
    verify=True,
)
for ent in manifest["entities"]:
    print(ent["name"], ent["type"], ent["primary"])
    for img in ent["images"]:
        print("   ", img["shot"], img["abs_path"], f'{img["width"]}x{img["height"]}')
```

---

## Plugging into an image-to-image / video workflow

The manifest is the contract. `see examples/consume_manifest.py` for a runnable
version of this:

```python
import json

run = json.load(open("runs/run-20260905-024816.json"))

refs = {}
for ent in run["entities"]:
    by_shot = {}
    for img in ent["images"]:
        by_shot.setdefault(img["shot"], []).append(img["abs_path"])
    refs[ent["slug"]] = {
        "name": ent["name"],
        "type": ent["type"],
        "primary": ent["primary"],     # single best frame
        "shots": by_shot,              # {"portrait": [...], "fullbody": [...]}
    }

# e.g. identity reference for a character, environment plate for a location
face  = refs["sherlock-holmes"]["shots"]["portrait"][0]
body  = refs["sherlock-holmes"]["shots"]["fullbody"][0]
plate = refs["taj-mahal"]["shots"]["establishing"][0]
```

Folders are keyed by slug and reused across prompts, so a second prompt
mentioning Sherlock Holmes costs nothing — the pipeline sees the folder is already at
plan and skips it (`--force` overrides). That makes `refs/` a growing, shared
reference library rather than per-run scratch.

---

## Three things to make

Everything above — extract, plan, search, filter, verify, manifest — is shared.
What changes is what the references are handed to, picked with `--mode`:

| `--mode` | model | references go in as | gives you |
|---|---|---|---|
| `video` *(default)* | MiniMax H3 Ref2VA | up to nine `<Picture N>` slots | a short clip with audio |
| `image` | Qwen-Image-Edit 2511 · Flux.2 Klein · Qwen-Image 2512 | conditioning images / reference latents | a still |
| `edit` | Qwen-Image-Edit 2511 · Flux.2 Klein | your frame first, then the references | that frame, changed |

```bash
python -m prompt2ref --comfy --mode image "Zendaya on a Paris rooftop at blue hour"
python -m prompt2ref --comfy --mode edit --edit-image shot.png "Put a Chanel No 5 bottle on the ledge"
python -m prompt2ref --comfy "Sherlock Holmes walks out of the Taj Mahal"     # video
```

Shared settings that mean different things per mode default per mode too:
`--megapixels` (0.4 video / 1.0 image), `--aspect` (16:9 / 1:1), `--pack`
(`per-entity` / `primary`), `--steps` (20 / whatever the image backend wants).

---

## ComfyUI: MiniMax H3 Ref2VA

Prompt2Ref can hand its references straight to the **MiniMax H3 Ref2VA** template
(`templates/minimax_h3_ref2va.json`). One command goes prompt → references →
a ready-to-open ComfyUI workflow:

```bash
python -m prompt2ref --comfy --verify "Sherlock Holmes walks out of the Taj Mahal holding an iPhone 16 Pro"
```

```
workflow : ...\Prompt2Ref\workflows\ref2va-20260905-033207.json
video    : 864x480  (0.41 MP)  39 frames  ~1.62s @ 24fps
pictures : 3 slots
   <Picture 1>  p2r_01_sherlock-holmes.png       1024x958   Sherlock Holmes
   <Picture 2>  p2r_02_taj-mahal.png        886x1024   Taj Mahal
   <Picture 3>  p2r_03_iphone-16-pro.png    1024x425   iPhone 16 Pro
installed: ...\ComfyUI\user\default\workflows\ref2va-20260905-033207.json
```

Open it from ComfyUI's workflow list and hit Run. What the builder does:

1. **Packs** each entity's references into a picture file in `ComfyUI/input/prompt2ref/`.
2. **Fills the slots** — points `<Picture N>`'s LoadImage at that file and
   un-bypasses the group. The template ships with all nine picture groups (and
   the `<Video N>` / `<Audio N>` groups) bypassed, i.e. in plain text-to-video
   mode; only the slots actually used are switched on.
3. **Rewrites the prompt** into MiniMax H3's sectioned form
   (`subject_definitions` / `summary` / `retention_analysis` /
   `detailed_description` / `overall_soundscape` / `non_diegetic_music`), with
   each entity bound to its `<Picture N>` tag and referenced by tag in the shot.
4. **Sets the video** — duration, and a resolution solved from a megapixel
   budget snapped to multiples of 32 (0.4 MP 16:9 → 864x480, matching the
   template's own size table). It also bypasses the template's half-resolution
   downscale so the number you ask for is the number that renders.

### Packing modes (`--pack`)

| mode | result |
|---|---|
| `per-entity` *(default)* | one slot per entity; that entity's shots tiled into one justified sheet |
| `primary` | one slot per entity, best single frame, no tiling |
| `per-image` | every image gets its own slot — highest fidelity, fills the nine slots fast |
| `all-in-one` | every entity on a single labelled sheet in `<Picture 1>` |

### Reference image size

`--ref-max 1024` is the default: the **delivered** file (sheet included) is
capped at 1024px on its longest edge.

```bash
--ref-max 2048          # bigger cap
--ref-native            # no resampling at all
--ref-megapixels 1.5    # cap by total pixels instead
--scale-per-tile        # apply the cap to each tile, letting the sheet grow past it
--labels / --no-labels  # entity names on sheets (on by default only for all-in-one)
```

### Video settings

```bash
--seconds 1             # duration (default 1)
--megapixels 0.4        # resolution budget (default 0.4)
--aspect 16:9           # 9:16, 1:1, 4:3, 21:9, or w:h
--frames 22             # pin the frame count instead of deriving it
--half-res              # keep the template's 0.5x downscale enabled
--keep-av               # leave the <Video N> / <Audio N> groups alone
```

**On "1 second":** the template derives frames as `max(5, round(s*24))` rounded
up to the next value congruent to 5 (mod 17), so `--seconds 1` yields
**39 frames (~1.6s at 24fps)**. The valid lengths near one second are 22
(~0.92s) and 39 (~1.63s) — use `--frames 22` for the closer one. The builder
prints this whenever the requested and actual durations diverge.

### Building without re-fetching

```bash
python -m prompt2ref --comfy-only                      # newest run
python -m prompt2ref --comfy-only --manifest runs/run-....json --pack all-in-one
python -m prompt2ref --comfy-only --aspect 9:16 --megapixels 0.6 --frames 22
```

The web UI drives the whole chain — see **The UI** below.

**Python**

```python
from prompt2ref import build_references
from prompt2ref.comfy import BuildOptions, ScalePolicy, build_workflow
from prompt2ref.config import load_settings

settings = load_settings()
manifest = build_references("...", verify=True)
result = build_workflow(manifest, settings, BuildOptions(
    pack="per-entity", scale=ScalePolicy(max_edge=1024),
    seconds=1.0, megapixels=0.4, aspect="16:9",
))
print(result.summary())
```

ComfyUI's install directory is auto-detected at `Desktop/AI/ComfyUI`; override
with `--comfy-root` or `PROMPT2REF_COMFY_ROOT`. If it isn't found, the images
are staged under `comfy_staging/input/` and the builder says so.

> The builder writes a **UI workflow** to open in ComfyUI, not an API-format
> graph — this template leans on SetNode/GetNode, subgraphs and rgthree group
> bypassers, and a UI→API conversion of that graph would not be trustworthy.

### Rendering headlessly (`--render`)

`--render` submits the job to a running ComfyUI and waits for the file:

```bash
python -m prompt2ref --comfy --render "Sherlock Holmes walks out of the Taj Mahal holding an iPhone 16 Pro"
python -m prompt2ref --comfy-only --render --steps 20 --seed 42 --ref-size max
```

Because `/prompt` only accepts API-format graphs, the runner does **not** post
the UI workflow. It rebuilds the executed path — UNETLoader → CLIPLoader
(`type: minimax`) → `MiniMaxH3ReferenceToVideo` → SamplerCustomAdvanced
(`res_multistep` / `simple` / 20 steps, matching the template) → VAEDecode +
VAEDecodeAudio → CreateVideo → SaveVideo — from the same packed references,
prompt and video settings. Model files are matched against what the server
actually reports from `/object_info`, so renamed checkpoints still resolve.
The UI workflow is still written for opening by hand; the two stay in sync.

```bash
--server http://127.0.0.1:8188   # ComfyUI address
--steps 20 --seed 0              # sampler
--ref-size match|max             # 'max' = 2048px short edge, much sharper identity, much slower
--render-timeout 7200
```

---

## ComfyUI: images and edits

Same chain, different tail. There is no shipped template here — an image model
wires its references its own way, so `--mode image` builds the graph from
scratch for whichever backend you pick and writes both the runnable graph and a
workflow you can open.

```bash
python -m prompt2ref --comfy --mode image --verify \
    "Zendaya on a Paris rooftop at blue hour, a Chanel No 5 bottle on the ledge beside her"
```

```
workflow : ...\Prompt2Ref\workflows\p2r-image-qwen-image-edit-20260913-1712.json
backend  : Qwen-Image-Edit 2511 (reference generate + edit)
image    : 1344x752  (1.01 MP)  x1  4 steps  cfg 1.0  seed 0
models   : qwen_image_edit_2511_nvfp4 + qwen_2.5_vl_7b_nvfp4 + qwen_image_vae + Lightning-4steps
pictures : 2 references
   ref 1  p2r_01_zendaya.png            683x1024   Zendaya
   ref 2  p2r_02_chanel-no-5.png        1024x768   Chanel No 5
installed: ...\ComfyUI\user\default\workflows\p2r-image-qwen-image-edit-20260913-1712.json
```

### Backends (`--backend`, `--list-backends`)

| backend | reference slots | edit | how the references get in |
|---|---|---|---|
| `qwen-image-edit` *(default)* | 3 | yes | `TextEncodeQwenImageEditPlus` image1/2/3 — tagged `Picture N` in the prompt, so a name in the text lands on a specific reference |
| `flux2-klein` | 4 | yes | VAE-encoded and chained onto the conditioning as `ReferenceLatent`s. Klein 9B needs the **Qwen3-8B** text encoder — the 4B loads fine and then dies in the sampler with a `7680` vs `12288` matmul |
| `qwen-image` | 0 | no | not at all — the references only shape the written prompt. The control case |

Model files are matched by substring against what your ComfyUI actually
reports, so a requantised or renamed checkpoint still resolves; a backend whose
files are missing says which one it wanted. Each backend brings its own step
count, cfg and sampling shift, and uses its step-distillation LoRA when one is
installed (`--no-turbo` to sample at full steps instead).

### The written prompt

`--mode image` rewrites the free prompt into one image prompt with every
reference bound by name, plus a negative prompt. `--mode edit` rewrites it into
an *instruction* about the existing frame instead — what to change, and
explicitly what to leave alone — with the references saying what the replaced
element must look like. Both fall back to a deterministic layout with no API key.

```bash
--style "35mm film still, shallow depth of field"   # appended to the prompt
--negative "..."                                    # override the negative
```

### Editing

```bash
python -m prompt2ref --comfy --mode edit --edit-image shot.png \
    "Replace the drink on the table with a bottle of Chanel No 5"
```

The frame you name is copied into `ComfyUI/input/prompt2ref_edit/` (its own
folder, because the packed-reference folder is wiped at the start of every
build) and takes the **first** conditioning slot — so a three-slot backend has
two left for references, and the prompt writer is told to start its tags at 2.
Output size follows the source: the edited frame is clamped to `--megapixels`
and sampled from its own latent, so framing survives. `--denoise` below 1 keeps
more of the original.

Passing `--edit-image` on its own implies `--mode edit`.

### Image settings

```bash
--megapixels 1.0      # resolution budget, each side rounded up to a multiple of 16
--aspect 16:9         # ignored in edit mode - the source decides
--batch 4             # images per run
--steps / --cfg       # override the backend's own defaults
--seed 42
--pack primary        # default for images: one best frame per entity, no tiling
```

`--pack primary` is the image default on purpose. Nine H3 slots can afford a
tiled contact sheet per entity; three cannot, and an image model conditioned on
a grid has a habit of drawing the grid. Use `--pack per-entity` when you want
the extra angles and can live with that.

`--render` works exactly as it does for video — build, submit, wait — and
streams ComfyUI's own latent previews while it samples.

---

## The UI

```bash
python -m prompt2ref --serve        # http://127.0.0.1:8077
```

One page, one button that runs the whole chain, and a **Video / Image / Edit**
switch at the top that decides what that button makes:

0. **What are you making** — the mode switch. Picking *Image* or *Edit* reveals
   the image-model picker (each backend labelled with its slot count, whether it
   can edit, and whether its files were actually found on this machine); *Edit*
   also reveals the file picker for the frame you want changed, which is staged
   into ComfyUI's input folder as soon as you choose it.
1. **Prompt** — write it with the entities named. Toggles for Claude NER,
   visual QC, refetch, and reusing cached references.
2. **Entities** — what was detected, editable before anything downloads (name,
   type, disambiguator; add/remove rows). The fetched references appear here as
   thumbnails as they land, each deletable.
3. **Settings** — the controls that apply to the current mode, and only those.
   Shared: aspect, megapixels, packing mode, reference size cap, steps, seed.
   Video adds length in seconds (or a frame override), reference detail
   (`match` / `max`), half-res and the live-preview options; image adds the
   batch size, cfg, style clause, negative prompt and the turbo toggle; edit
   adds denoise. Mode defaults are applied on the switch, so *Image* jumps to
   1.0 MP / 1:1 / `primary` without you touching anything.

   A live readout mirrors the server's own maths — `864×480 (0.41 MP) · 39
   frames ≈ 1.62s @ 24fps · quantised from 1s (17k+5 rule)` for video, or
   `1344×752 (1.01 MP) · 4 steps · cfg 1 · qwen-image-edit + turbo` for an
   image — so the numbers you'll actually get are visible *before* you spend
   four minutes on a render.
4. **Output** — the sampling preview streams in as the model works, then the
   result replaces it: a `<video>` player for a clip, the image for a still
   (with the rest of a batch as thumbnails underneath), both with a download
   link and the on-disk path. A progress bar, step counter, seconds-per-step
   and elapsed time sit underneath, plus a cancel button, the written prompt
   and negative, and the packed reference sheets.

### Prefilling from the URL

Every control can be set from the query string, and `run=1` starts the whole
chain on load — handy for handing someone a link that renders itself:

```
http://127.0.0.1:8077/?prompt=Zendaya%20on%20a%20Paris%20rooftop...&seconds=10
    &aspect=16:9&mp=0.4&pack=per-entity&ref=1024&refdetail=match
    &steps=20&seed=11&preview=1&pvframes=6&verify=1&run=1

http://127.0.0.1:8077/?mode=image&backend=qwen-image-edit&prompt=...
    &aspect=16:9&mp=1&pack=primary&batch=2&style=35mm%20film%20still&run=1
```

Keys: `mode`, `backend`, `prompt`, `seconds`, `frames`, `aspect`, `mp`, `pack`,
`ref`, `refdetail`, `steps`, `seed`, `pvframes`, `preview`, `labels`, `batch`,
`cfg`, `denoise`, `style`, `negative`, `turbo`, `verify`, `force`, `run`.
`mode` is applied first, so anything after it overrides that mode's defaults.
(`edit` can be prefilled, but the frame itself still has to be chosen by hand.)

### Live preview

The preview comes from KJNodes' **`ModelPreviewOverrideKJ`**, inserted between
the UNET loader and the sampler. It decodes each sampler step with the tiny
`taeh3.safetensors` VAE and pushes it over ComfyUI's websocket as a
`kj_preview_override` message. With `preview_frames > 1` the node encodes a
short animated clip per step (NVENC mp4 when available, animated WebP
otherwise), so you watch the *motion* form, not just one frame.

The browser can't talk to ComfyUI directly — its `origin_only` middleware
rejects a page on :8077 calling :8188, because the Host and Origin ports differ.
So the Prompt2Ref server relays: it opens the websocket itself (server-to-server
requests carry no `Origin`), subscribes with the same `client_id` it submits
under, and re-streams everything to the page as NDJSON. Rendered files are
served back through `/api/output`, restricted to the ComfyUI output tree.

Image renders have no KJ override in the way, so they stream ComfyUI's **own**
binary websocket previews instead — the relay decodes those to the same
`preview` event the page already knows how to draw.

| endpoint | purpose |
|---|---|
| `POST /api/extract` | prompt → entities |
| `POST /api/fetch` | entities → images (NDJSON progress) |
| `POST /api/comfy` | build the video workflow only |
| `POST /api/render` | build **and** render the video (NDJSON: `build`, `submitted`, `preview`, `progress`, `result`) |
| `GET /api/image/backends` | the image backends, with whether each one's model files were found |
| `POST /api/image/source` | stash the frame to edit (data URL in, ComfyUI input name out) |
| `POST /api/image` | build the image / edit workflow only |
| `POST /api/image/render` | build **and** render an image or an edit (same NDJSON shape) |
| `POST /api/render/cancel` | interrupt the running job |
| `GET /api/output?path=` | serve a rendered file |

---

## The ComfyUI node pack

Everything above runs Prompt2Ref *outside* ComfyUI and hands it a finished
workflow. `comfyui_prompt2ref/` is the other direction: the same pipeline as
ComfyUI nodes, so the prompt goes in at the top of a graph and the references
are collected as part of the run.

```bash
# from ComfyUI/custom_nodes — a symlink keeps it in sync with the checkout
mklink /D comfyui_prompt2ref C:\Users\...\Prompt2Ref\comfyui_prompt2ref     # Windows
ln -s /path/to/Prompt2Ref/comfyui_prompt2ref .                              # elsewhere
```

Then restart ComfyUI. The nodes import `prompt2ref` from the checkout (found via
`PROMPT2REF_REPO`, the symlink's own parent, or the default path), so there is
one copy of the pipeline, not two.

| node | in → out |
|---|---|
| **P2R Provider: Local CLIP** / **Anthropic API** | → `provider` — every LLM step takes this, so local or hosted is one rewire |
| **P2R Extract Entities** | prompt → `entities_json` |
| **P2R Entity Slot** ×5 + **Combine** | `entities_json` → hand-edited, individually switchable entities |
| **P2R Fetch References** | entities → `manifest_json`, images on disk |
| **P2R Verify References** | manifest → manifest, watermarked / off-subject frames deleted |
| **P2R Pack References** | manifest → `ref_image_1..9` for MiniMax H3 |
| **P2R Pack References (image slots)** | manifest → `ref_image_1..4`, entities past the last slot folded onto it rather than dropped |
| **P2R Build H3 Prompt** | captions → the sectioned H3 prompt |
| **P2R Build Image / Edit Prompt** | captions → `image_prompt`, `negative_prompt` |
| **P2R Image Resolution** | megapixels + aspect → `width`, `height` |

### The demo workflows

Four saved workflows, one per thing the pack makes, in `workflows/` and
installed into ComfyUI's own list:

| workflow | what it does |
|---|---|
| `p2r_video_minimax_h3` | prompt → references → MiniMax H3 Ref2VA clip (VDN-H3 hybrid attention + its 8-step turbo adapter) |
| `p2r_image_qwen_edit_generate` | prompt → references → Qwen-Image-Edit still, 3 conditioning slots |
| `p2r_image_qwen_edit_edit` | your frame (a `LoadImage` to point at your own file) + references → an edit of it |
| `p2r_image_flux2_klein_generate` | prompt → references → Flux.2 Klein still, references as chained reference latents |

None of them is pre-baked: change the prompt at the top and the whole chain
re-runs, collecting different references.

On the video demo specifically: VDN-H3's `branch_weights: stream` streams its
~2.2 GB int8 branch from disk per block per step, which is the right trade on a
card short of VRAM but needs system RAM to page against. On 16 GB with about
1 GB free it stops making progress entirely — no sampler step for a minute and a
half — rather than failing outright. Free some memory first (ComfyUI's own
*Unload models* helps), or rewire `scheduler` and `guider` from `vdn` to the
community `lora` node already sitting in the graph, which skips the streaming
branch. 608×352 × 56 frames through the LoRA branch renders comfortably.

Rebuild them with

```bash
python comfyui_prompt2ref/build_demo_workflows.py
```

Each demo also carries both providers — the Anthropic one wired up, the local
CLIP one sitting beside it unwired, as the zero-cost swap.

---

## Layout

```
prompt2ref/
  config.py      settings, .env loading, source auto-detection
  entities.py    Entity + EntityType, slugging, dedupe
  extract.py     Claude NER (structured output) + heuristic fallback
  shots.py       per-type shot plans — the "portrait AND full body" logic
  sources/       ddg · wikimedia · openverse · serpapi · google_cse
  download.py    fetch, validate, watermark blocklist, dhash dedupe, save
  verify.py      optional Claude vision QC pass
  pipeline.py    orchestration, caching, manifest + index
  cli.py         command line
  server.py      FastAPI app
  static/        the web UI
  comfy/
    template.py    read/write the Ref2VA workflow (slots, prompt, resolution)
    imagepack.py   scale policy + justified contact sheets (shared by both paths)
    promptgen.py   free prompt -> MiniMax H3 sectioned prompt
    builder.py     the video assembler
    imagegraph.py  the image backends: model needles + one graph builder each
    imageprompt.py free prompt -> image prompt / edit instruction + negative
    imagebuild.py  the image assembler
    uigraph.py     API graph -> an openable ComfyUI workflow
    schemas.json   bundled node schemas, so that works with no server running
    run.py         API-graph submit + websocket live-preview relay
comfyui_prompt2ref/       <- symlink or copy this into ComfyUI/custom_nodes
  nodes_provider.py   local CLIP or Anthropic API, behind one P2R_PROVIDER type
  nodes_extract.py    prompt -> entities
  nodes_slots.py      five switchable, hand-editable entity slots
  nodes_fetch.py      entities -> references on disk
  nodes_verify.py     visual QC, by whichever provider is wired in
  nodes_pack.py       references -> nine H3 picture slots
  nodes_h3prompt.py   references -> the H3 sectioned prompt
  nodes_image.py      references -> image slots, image/edit prompt, resolution
  build_demo_workflows.py   writes workflows/p2r_*.json
templates/
  minimax_h3_ref2va.json
workflows/
  p2r_video_minimax_h3.json            the demos, also installed into ComfyUI's
  p2r_image_qwen_edit_generate.json    saved workflow list
  p2r_image_qwen_edit_edit.json
  p2r_image_flux2_klein_generate.json
examples/
  consume_manifest.py
```

## Notes and limits

- Search results are the open web. The blocklist and the optional `--verify`
  pass catch most watermarked or off-subject hits, but nothing here checks
  licensing for you — `entity.json` records the source page and any license
  Wikimedia/Openverse reported so you can.
- DuckDuckGo rate-limits bursts; entity fetches run 3-wide by default
  (`--workers`) and each search retries with backoff.
- `--verify` costs one extra vision call per entity (images are downscaled to
  512px first) and roughly doubles wall-clock time for a run. Use it whenever
  the references are headed for a generation run: without it, movie posters
  with burned-in titles and upscaler watermarks do get through.
- `--verify` only inspects **newly downloaded** frames. Entities served from
  cache pass through untouched — re-run with `--force --verify` to re-check a
  folder collected before the check existed.
- Image models take three or four references, not nine. Name more entities than
  that and the extras are folded onto the last sheet (nodes) or dropped with a
  warning (CLI/UI) — `--pack all-in-one` is the deliberate way to get everything
  into one slot.
- `--mode image` writes its graph from scratch rather than editing a template,
  so it can be POSTed to `/prompt` directly. The openable workflow beside it is
  converted from that graph using each node class's declared input order, read
  from a running ComfyUI when there is one and from `comfy/schemas.json`
  otherwise. A node class in neither is the one case where the workflow file is
  skipped — the graph itself still renders.
- The image backends are wired for the checkpoints on *this* machine and matched
  by substring. A different quantisation is fine; a different model family needs
  a new entry in `comfy/imagegraph.py` — each one is a ~30-line function.
