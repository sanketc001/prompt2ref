"""How a downstream image-to-image / image-to-video workflow reads a run.

    python examples/consume_manifest.py                    # newest run
    python examples/consume_manifest.py runs/run-....json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def latest_manifest() -> Path:
    runs = sorted((ROOT / "runs").glob("run-*.json"))
    if not runs:
        raise SystemExit("no runs yet - try: python -m prompt2ref \"your prompt\"")
    return runs[-1]


def load_refs(manifest_path: Path) -> dict:
    """-> {slug: {name, type, primary, shots: {shot_key: [abs paths]}}}"""
    run = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = {}
    for ent in run["entities"]:
        by_shot: dict[str, list[str]] = {}
        for img in ent.get("images", []):
            by_shot.setdefault(img["shot"], []).append(img["abs_path"])
        refs[ent["slug"]] = {
            "name": ent["name"],
            "type": ent["type"],
            "primary": ent.get("primary"),
            "shots": by_shot,
        }
    return refs


def reference_set(refs: dict) -> list[dict]:
    """Pick the frames a generation call should actually be conditioned on.

    People get identity (face) + silhouette (full body); places get the
    establishing plate; everything else gets its primary frame.
    """
    picks = []
    for slug, r in refs.items():
        shots = r["shots"]
        if r["type"] in ("person", "fictional_character"):
            wanted = [("identity", "portrait"), ("body", "fullbody")]
        elif r["type"] in ("landmark", "place"):
            wanted = [("environment", "establishing")]
        elif r["type"] == "organization":
            wanted = [("logo", "logo")]
        else:
            wanted = [("subject", next(iter(shots), ""))]
        for role, key in wanted:
            if shots.get(key):
                picks.append({"slug": slug, "name": r["name"], "role": role,
                              "path": shots[key][0]})
    return picks


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_manifest()
    refs = load_refs(path)
    print("manifest:", path)
    for slug, r in refs.items():
        counts = ", ".join(f"{k}x{len(v)}" for k, v in r["shots"].items()) or "empty"
        print(f"  {r['name']:<20} {r['type']:<20} {counts}")
    print("\nreference set for generation:")
    for p in reference_set(refs):
        print(f"  [{p['role']:<11}] {p['name']:<20} {p['path']}")
