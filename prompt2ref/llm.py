"""Shared call wrapper for the three spots that ask Claude for a JSON-schema reply.

Not every model in an account's catalog supports adaptive thinking or the
`effort` knob - older dated snapshots (e.g. claude-haiku-4-5-20251001) reject
one or both with a 400. Since the UI lets a user pick any model the API lists,
degrade gracefully instead of falling all the way back to the heuristic path.
"""

from __future__ import annotations

import anthropic


def structured_message(client: "anthropic.Anthropic", *, model: str, max_tokens: int,
                        system: str, schema: dict, messages: list,
                        effort: str | None = None):
    output_config = {"format": {"type": "json_schema", "schema": schema}}
    if effort:
        output_config["effort"] = effort
    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  thinking={"type": "adaptive"}, output_config=output_config,
                  messages=messages)
    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as exc:
        if "adaptive thinking is not supported" not in str(exc):
            raise
        kwargs.pop("thinking")
        try:
            return client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc2:
            if not effort or "does not support the effort parameter" not in str(exc2):
                raise
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
            return client.messages.create(**kwargs)
