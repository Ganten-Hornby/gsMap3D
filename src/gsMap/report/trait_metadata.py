"""Presentation metadata and collision-free, offline trait payloads."""

import json


def trait_payload_js(registry, trait, payload_json):
    return (
        f"window.{registry}=window.{registry}||Object.create(null);\n"
        f"window.{registry}[{json.dumps(trait)}]={payload_json};"
    )


def trait_payload_current(path, registry):
    try:
        with path.open("rb") as handle:
            return handle.read(160).startswith(f"window.{registry}=".encode())
    except OSError:
        return False
