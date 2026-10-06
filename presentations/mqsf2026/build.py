# Copyright (c) 2026 Munich Quantum Software Company GmbH
# SPDX-License-Identifier: MIT

# /// script
# dependencies = ["pygments>=2.19,<3", "mlir-pygments==1.0.0", "openqasm-pygments>=0.2,<0.3"]
# ///
"""Package recorded MQSF demonstrations without loading native MQT Core."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from capture_execution import validate_execution
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

HERE = Path(__file__).resolve().parent


def encode_data(data: dict[str, Any]) -> str:
    """Encode JSON for an inline script without allowing HTML termination.

    Returns:
        JSON with HTML opening delimiters escaped.
    """
    return json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def validate_capture(data: dict[str, Any]) -> None:
    """Reject inconsistent topology, execution evidence, and ambiguous controls.

    Raises:
        ValueError: If captured evidence contradicts itself.
    """
    if data.get("schema_version") != 1 or not data.get("provenance", {}).get("core_revision"):
        msg = "Capture must identify its schema and Core revision"
        raise ValueError(msg)
    sites = {site["id"] for site in data["device"]["sites"]}
    if len(sites) != len(data["device"]["sites"]):
        msg = "Duplicate topology site"
        raise ValueError(msg)
    if any(len(edge) != 2 or any(site not in sites for site in edge) for edge in data["device"]["edges"]):
        msg = "Invalid topology edge"
        raise ValueError(msg)
    scenario_ids: set[str] = set()
    for scenario in data["scenarios"]:
        if scenario["id"] in scenario_ids:
            msg = "Duplicate scenario ID"
            raise ValueError(msg)
        scenario_ids.add(scenario["id"])
        variant_ids: set[str] = set()
        for variant in scenario["variants"]:
            if variant["id"] in variant_ids:
                msg = "Duplicate variant ID"
                raise ValueError(msg)
            variant_ids.add(variant["id"])
            for field in ("stages", "exports"):
                artifacts = variant.get(field, [])
                if len({artifact["id"] for artifact in artifacts}) != len(artifacts):
                    msg = f"Duplicate {field} ID"
                    raise ValueError(msg)
                for artifact in artifacts:
                    if "code" not in artifact or "sha256" not in artifact:
                        continue
                    digest = hashlib.sha256(artifact["code"].encode()).hexdigest()
                    if artifact["sha256"] != digest:
                        msg = "Artifact hash differs from its original source"
                        raise ValueError(msg)
            hashes = {
                artifact["id"]: hashlib.sha256(artifact["code"].encode()).hexdigest()
                for artifact in variant.get("exports", [])
                if "code" in artifact
            }
            executions = [(None, variant.get("execution")), *variant.get("executions", {}).items()]
            for export_id, execution in executions:
                if not execution or execution.get("unavailable_reason"):
                    continue
                validate_execution(execution)
                expected = hashes.values() if export_id is None else [hashes.get(export_id)]
                if execution["payload_sha256"] not in expected:
                    msg = "Executed payload does not match its captured export"
                    raise ValueError(msg)


def build(output: Path, capture: Path = HERE / "captures/demo.json.gz", source: Path = HERE) -> Path:
    """Build a standalone HTML file and ZIP from verified capture fixtures.

    Returns:
        Path to the self-contained HTML presentation.

    Raises:
        ValueError: If captures or template markers are inconsistent.
    """
    raw = capture.read_bytes()
    data = json.loads(gzip.decompress(raw) if capture.suffix == ".gz" else raw)
    validate_capture(data)
    formatter = HtmlFormatter(cssclass="highlight", nowrap=False, style="monokai")
    for scenario in data["scenarios"]:
        for variant in scenario["variants"]:
            for artifact in [*variant.get("stages", []), *variant.get("exports", [])]:
                if "code" in artifact:
                    code = artifact["code"]
                    artifact["sha256"] = hashlib.sha256(code.encode()).hexdigest()
                    lines = code.splitlines(keepends=True)
                    artifact["line_count"] = len(lines)
                    if len(lines) > 200:
                        artifact["full_code_gzip"] = base64.b64encode(gzip.compress(code.encode(), mtime=0)).decode(
                            "ascii"
                        )
                        start = max(0, min(artifact.get("focus_line", 1) - 5, len(lines) - 200))
                        excerpt = lines[start : start + 200]
                        artifact["code"] = "".join(excerpt)
                        artifact["excerpt_lines"] = len(excerpt)
                        artifact["excerpt_start_line"] = start + 1
                    language = artifact.get("language", "text")
                    lexer = get_lexer_by_name("openqasm3" if language == "qasm" else language)
                    artifact["html"] = highlight(artifact["code"], lexer, formatter)
    css = source.joinpath("presentation.css").read_text(encoding="utf-8")
    css = formatter.get_style_defs(".highlight") + "\n" + css
    script = source.joinpath("presentation.js").read_text(encoding="utf-8")
    html = source.joinpath("index.html").read_text(encoding="utf-8")
    substitutions = {
        "<!-- MQSF_STYLES -->": f"<style>{css}</style>",
        "<!-- MQSF_DATA -->": f"<script>window.MQSF_DATA={encode_data(data)};</script>",
        "<!-- MQSF_SCRIPT -->": f"<script>{script}</script>",
    }
    for marker, replacement in substitutions.items():
        if html.count(marker) != 1:
            msg = f"Expected exactly one template marker: {marker}"
            raise ValueError(msg)
        html = html.replace(marker, replacement)
    output.mkdir(parents=True, exist_ok=True)
    result = output / "index.html"
    result.write_text(html, encoding="utf-8")
    with zipfile.ZipFile(output / "mqsf-2026.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result, "index.html")
    return result


def main() -> None:
    """Package committed captures; native execution is a separate command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "build/mqsf2026")
    parser.add_argument("--capture", type=Path, default=HERE / "captures/demo.json.gz")
    args = parser.parse_args()
    print(build(args.output, args.capture))


if __name__ == "__main__":
    main()
