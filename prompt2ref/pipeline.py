"""Orchestration: prompt -> entities -> shots -> images on disk -> manifest."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings, load_settings
from .download import FetchContext, download_shot
from .entities import Entity, EntityType
from .extract import extract_entities
from .shots import Shot, expand_queries, plan_for
from .sources import search_all
from .verify import verify_images

log = logging.getLogger("prompt2ref.pipeline")

ENTITY_FILE = "entity.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._lock = threading.Lock()

    # -- step 1 -------------------------------------------------------------
    def extract(self, prompt: str, use_llm: bool = True) -> list[Entity]:
        return extract_entities(prompt, self.settings, use_llm=use_llm)

    # -- step 2 -------------------------------------------------------------
    def collect_entity(
        self,
        entity: Entity,
        force: bool = False,
        on_event=None,
        per_shot: int | None = None,
    ) -> dict:
        """Fill one entity's folder to plan, reusing whatever is already there."""
        s = self.settings
        dest = s.entity_dir(entity.slug)
        record = _load_entity_record(dest)
        if force:
            existing = {}
            _clear_images(dest)  # otherwise the old files linger, orphaned by the new record
        else:
            existing = _index_existing(record, dest)

        shots = plan_for(entity.type, per_shot if per_shot is not None else s.per_shot)
        ctx = FetchContext()
        for imgs in existing.values():
            for i in imgs:
                if i.get("sha256") and i.get("dhash"):
                    ctx.remember(i["sha256"], i["dhash"])

        new_by_shot: dict[str, list] = {}
        for shot in shots:
            have = existing.get(shot.key, [])
            missing = shot.count - len(have)
            if missing <= 0:
                self._emit(on_event, {"type": "shot_cached", "entity": entity.slug,
                                      "shot": shot.key, "have": len(have)})
                continue

            self._emit(on_event, {"type": "shot_start", "entity": entity.slug,
                                  "shot": shot.key, "label": shot.label, "want": missing})
            new_by_shot[shot.key] = self._fill_shot(
                entity, shot, dest, ctx, missing,
                start_index=_next_index(dest, shot.key), on_event=on_event,
            )
            if not new_by_shot[shot.key] and shot.required:
                self._emit(on_event, {"type": "shot_empty", "entity": entity.slug,
                                      "shot": shot.key})

        if s.verify:
            self._verify_and_refill(entity, shots, dest, ctx, new_by_shot, on_event)

        images: list[dict] = []
        for shot in shots:
            images.extend(existing.get(shot.key, []))
            images.extend(i.to_dict(s.refs_dir) for i in new_by_shot.get(shot.key, []))

        out = {
            **entity.to_dict(),
            "dir": str(dest.relative_to(s.refs_dir.parent)).replace("\\", "/"),
            "abs_dir": str(dest),
            "updated_at": _now(),
            "shots": [{"key": sh.key, "label": sh.label, "want": sh.count} for sh in shots],
            "images": images,
            "primary": _pick_primary(images, shots),
        }
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ENTITY_FILE).write_text(json.dumps(out, indent=2), encoding="utf-8")
        self._emit(on_event, {"type": "entity_done", "entity": entity.slug,
                              "count": len(images)})
        return out

    def _fill_shot(self, entity: Entity, shot: Shot, dest: Path, ctx: FetchContext,
                   want: int, start_index: int, on_event) -> list:
        s = self.settings
        saved: list = []
        idx = start_index
        for query in self._queries_for(shot, entity):
            if len(saved) >= want:
                break
            limit = max(s.min_candidates, (want - len(saved)) * s.candidates_multiplier)
            cands = search_all(query, limit, s)
            self._emit(on_event, {"type": "search", "entity": entity.slug, "shot": shot.key,
                                  "query": query, "hits": len(cands)})
            if not cands:
                continue
            got = download_shot(
                cands, shot, dest, s, ctx,
                want=want - len(saved), start_index=idx, query=query,
                on_event=lambda ev: self._emit(on_event, {**ev, "entity": entity.slug}),
            )
            saved.extend(got)
            idx += len(got)
        return saved

    def _verify_and_refill(self, entity: Entity, shots, dest: Path, ctx: FetchContext,
                           new_by_shot: dict[str, list], on_event) -> None:
        """Show the new frames to Claude, delete the rejects, refill the holes once."""
        fresh = [i for lst in new_by_shot.values() for i in lst]
        if not fresh:
            return
        verdicts = verify_images(entity, fresh, self.settings)
        if not verdicts:
            return

        by_key = {sh.key: sh for sh in shots}
        holes: dict[str, int] = {}
        for key, items in new_by_shot.items():
            kept = []
            for item in items:
                ok, reason = verdicts.get(item.path.name, (True, ""))
                if ok:
                    kept.append(item)
                    continue
                item.path.unlink(missing_ok=True)
                holes[key] = holes.get(key, 0) + 1
                self._emit(on_event, {"type": "rejected", "entity": entity.slug,
                                      "shot": key, "file": item.path.name,
                                      "reason": reason})
            new_by_shot[key] = kept

        for key, missing in holes.items():
            shot = by_key.get(key)
            if not shot:
                continue
            self._emit(on_event, {"type": "refill", "entity": entity.slug,
                                  "shot": key, "want": missing})
            new_by_shot[key].extend(
                self._fill_shot(entity, shot, dest, ctx, missing,
                                start_index=_next_index(dest, key), on_event=on_event)
            )

    @staticmethod
    def _queries_for(shot: Shot, entity: Entity) -> list[str]:
        """Planned queries first, then the bare name as a wider fallback pass.

        The descriptor sharpens a query but also narrows it; dropping it is the
        cheapest way to recover recall when a shot comes up short.
        """
        queries = expand_queries(shot, entity.query_base)
        for q in expand_queries(shot, entity.search_name):
            if q not in queries:
                queries.append(q)
        return queries

    # -- step 3 -------------------------------------------------------------
    def run(
        self,
        prompt: str,
        entities: list[Entity] | None = None,
        use_llm: bool = True,
        force: bool = False,
        per_shot: int | None = None,
        on_event=None,
    ) -> dict:
        started = time.time()
        s = self.settings
        if entities is None:
            self._emit(on_event, {"type": "extract_start"})
            entities = self.extract(prompt, use_llm=use_llm)
        self._emit(on_event, {
            "type": "entities",
            "entities": [e.to_dict() for e in entities],
        })

        results: dict[str, dict] = {}
        if entities:
            workers = max(1, min(s.workers, len(entities)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self.collect_entity, e, force, on_event, per_shot): e
                    for e in entities
                }
                for fut in as_completed(futures):
                    e = futures[fut]
                    try:
                        results[e.slug] = fut.result()
                    except Exception as exc:
                        log.exception("entity %s failed", e.slug)
                        self._emit(on_event, {"type": "entity_error", "entity": e.slug,
                                              "error": str(exc)})
                        results[e.slug] = {**e.to_dict(), "images": [], "error": str(exc)}

        ordered = [results[e.slug] for e in entities if e.slug in results]
        manifest = {
            "prompt": prompt,
            "created_at": _now(),
            "elapsed_s": round(time.time() - started, 1),
            "refs_dir": str(s.refs_dir),
            "model": s.model if use_llm and s.api_key else "heuristic",
            "sources": list(s.sources),
            "entities": ordered,
            "total_images": sum(len(e.get("images", [])) for e in ordered),
        }
        path = self.write_manifest(manifest)
        manifest["manifest_path"] = str(path)
        self._emit(on_event, {"type": "done", "manifest_path": str(path),
                              "total_images": manifest["total_images"]})
        return manifest

    # -- persistence --------------------------------------------------------
    def write_manifest(self, manifest: dict) -> Path:
        s = self.settings
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = s.runs_dir / ("run-%s.json" % stamp)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (s.refs_dir / "latest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.rebuild_index()
        return path

    def rebuild_index(self) -> dict:
        """A flat catalogue of every entity folder, for reuse across prompts."""
        s = self.settings
        entries = []
        for d in sorted(p for p in s.refs_dir.iterdir() if p.is_dir()):
            rec = _load_entity_record(d)
            if not rec:
                continue
            entries.append({
                "slug": d.name,
                "name": rec.get("name", d.name),
                "type": rec.get("type", "other"),
                "images": len(rec.get("images", [])),
                "primary": rec.get("primary"),
                "updated_at": rec.get("updated_at"),
            })
        index = {"updated_at": _now(), "entities": entries}
        (s.refs_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        return index

    def _emit(self, on_event, payload: dict) -> None:
        if not on_event:
            return
        with self._lock:
            try:
                on_event(payload)
            except Exception:
                log.debug("event handler raised", exc_info=True)


def _clear_images(dest: Path) -> None:
    if not dest.is_dir():
        return
    for f in dest.iterdir():
        if f.is_file() and f.name != ENTITY_FILE:
            f.unlink(missing_ok=True)


def _next_index(dest: Path, shot_key: str) -> int:
    """First unused NN for `<shot>_NN.*`, so a refill never overwrites a keeper."""
    used = {
        int(f.stem.rsplit("_", 1)[1])
        for f in dest.glob(shot_key + "_*.*")
        if f.stem.rsplit("_", 1)[-1].isdigit()
    }
    n = 1
    while n in used:
        n += 1
    return n


def _load_entity_record(dest: Path) -> dict:
    f = dest / ENTITY_FILE
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _index_existing(record: dict, dest: Path) -> dict[str, list[dict]]:
    """Group the record's images by shot, dropping entries whose file is gone."""
    grouped: dict[str, list[dict]] = {}
    for img in record.get("images", []):
        name = img.get("file")
        if not name or not (dest / name).exists():
            continue
        grouped.setdefault(img.get("shot", "reference"), []).append(img)
    return grouped


def _pick_primary(images: list[dict], shots: tuple[Shot, ...]) -> str | None:
    """The single best frame to hand a workflow that only accepts one reference."""
    if not images:
        return None
    order = [s.key for s in shots]
    def rank(i: dict):
        key = i.get("shot", "")
        return (order.index(key) if key in order else 99, -(i.get("width", 0) * i.get("height", 0)))
    return sorted(images, key=rank)[0].get("path")


def build_references(prompt: str, **kwargs) -> dict:
    """One-call entry point: prompt in, manifest dict out.

    Any Settings field (refs_dir, verify, sources, min_width, ...) can be passed
    here; everything else goes to Pipeline.run (entities, force, on_event, ...).
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(Settings)}
    settings_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in fields}
    return Pipeline(load_settings(**settings_kwargs)).run(prompt, **kwargs)
