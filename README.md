# Prompt2Ref

Type a generation prompt, get a local folder of reference images for every named
entity in it — ready to feed a reference image-to-image / image-to-video workflow.

```
"Cinematic shot of Tom Cruise in front of the Taj Mahal holding an iPhone 16 Pro"

refs/
  tom-cruise/     portrait_01.jpg  portrait_02.jpg  fullbody_01.jpg  fullbody_02.jpg  profile_01.jpg
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
python -m prompt2ref "Tom Cruise in front of the Taj Mahal holding an iPhone 16 Pro"
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
    "Tom Cruise in front of the Taj Mahal holding an iPhone 16 Pro",
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
face  = refs["tom-cruise"]["shots"]["portrait"][0]
body  = refs["tom-cruise"]["shots"]["fullbody"][0]
plate = refs["taj-mahal"]["shots"]["establishing"][0]
```

Folders are keyed by slug and reused across prompts, so a second prompt
mentioning Tom Cruise costs nothing — the pipeline sees the folder is already at
plan and skips it (`--force` overrides). That makes `refs/` a growing, shared
reference library rather than per-run scratch.

---

## ComfyUI: MiniMax H3 Ref2VA

Prompt2Ref can hand its references straight to the **MiniMax H3 Ref2VA** template
(`templates/minimax_h3_ref2va.json`). One command goes prompt → references →
a ready-to-open ComfyUI workflow:

```bash
python -m prompt2ref --comfy --verify "Tom Cruise walks out of the Taj Mahal holding an iPhone 16 Pro"
```

```
workflow : ...\Prompt2Ref\workflows\ref2va-20260905-033207.json
video    : 864x480  (0.41 MP)  39 frames  ~1.62s @ 24fps
pictures : 3 slots
   <Picture 1>  p2r_01_tom-cruise.png       1024x958   Tom Cruise
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
python -m prompt2ref --comfy --render "Tom Cruise walks out of the Taj Mahal holding an iPhone 16 Pro"
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

## The UI

```bash
python -m prompt2ref --serve        # http://127.0.0.1:8077
```

One page, four steps, one **Generate video** button that runs the whole chain:

1. **Prompt** — write it with the entities named. Toggles for Claude NER,
   visual QC, refetch, and reusing cached references.
2. **Entities** — what was detected, editable before anything downloads (name,
   type, disambiguator; add/remove rows). The fetched references appear here as
   thumbnails as they land, each deletable.
3. **Settings** — aspect, video megapixels, length in seconds (or an explicit
   frame override), reference packing mode, reference size cap, reference detail
   (`match` / `max`), steps, seed, live-preview options. A live readout mirrors
   the server's own maths: `864×480 (0.41 MP) · 39 frames ≈ 1.62s @ 24fps ·
   quantised from 1s (17k+5 rule)` — so the quantisation is visible *before*
   you spend four minutes on a render.
4. **Output** — the sampling preview streams in as the model works, then the
   finished clip replaces it in a `<video>` player with download link and the
   on-disk path. A progress bar, step counter, seconds-per-step and elapsed time
   sit underneath, plus a cancel button, the generated H3 prompt, and the packed
   `<Picture N>` sheets.

### Prefilling from the URL

Every control can be set from the query string, and `run=1` starts the whole
chain on load — handy for handing someone a link that renders itself:

```
http://127.0.0.1:8077/?prompt=Zendaya%20on%20a%20Paris%20rooftop...&seconds=10
    &aspect=16:9&mp=0.4&pack=per-entity&ref=1024&refdetail=match
    &steps=20&seed=11&preview=1&pvframes=6&verify=1&run=1
```

Keys: `prompt`, `seconds`, `frames`, `aspect`, `mp`, `pack`, `ref`, `refdetail`,
`steps`, `seed`, `pvframes`, `preview`, `labels`, `verify`, `force`, `run`.

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

| endpoint | purpose |
|---|---|
| `POST /api/extract` | prompt → entities |
| `POST /api/fetch` | entities → images (NDJSON progress) |
| `POST /api/comfy` | build the workflow only |
| `POST /api/render` | build **and** render (NDJSON: `build`, `submitted`, `preview`, `progress`, `result`) |
| `POST /api/render/cancel` | interrupt the running job |
| `GET /api/output?path=` | serve a rendered file |

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
    imagepack.py   scale policy + justified contact sheets
    promptgen.py   free prompt -> MiniMax H3 sectioned prompt
    builder.py     the assembler
    run.py         API-graph submit + websocket live-preview relay
templates/
  minimax_h3_ref2va.json
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
