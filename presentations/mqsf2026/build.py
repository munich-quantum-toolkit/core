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
import re
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
    if targets := data.get("targets"):
        application = data.get("application", {})
        source = application.get("representative_source", {})
        source_hash = hashlib.sha256(source.get("code", "").encode()).hexdigest()
        provenance = data.get("target_provenance", {})
        if (
            not source.get("code")
            or source.get("sha256") != source_hash
            or provenance.get("source_sha256") != source_hash
        ):
            msg = "Target captures must identify the exact application source"
            raise ValueError(msg)
        if application.get("provenance", {}).get("core_revision") != data["provenance"]["core_revision"] or any(
            not provenance.get(key) or provenance[key] != data["provenance"].get(key)
            for key in ("compiler_sha256", "execution_script_sha256")
        ):
            msg = "Merged application and target provenance disagree"
            raise ValueError(msg)
        if len({target["id"] for target in targets}) != len(targets):
            msg = "Duplicate target ID"
            raise ValueError(msg)
        common_stages: dict[str, dict] = {}
        for target in targets:
            compilation = target["compilation"]
            stages = compilation.get("stages", [])
            stage_ids = [stage["id"] for stage in stages]
            if (
                stage_ids[:4] != ["source", "qc", "qco", "optimized"]
                or "place-and-route" not in stage_ids
                or "target-native-synthesis" not in stage_ids
                or stage_ids.index("place-and-route") > stage_ids.index("target-native-synthesis")
                or any(not stage.get("code") or not stage.get("sha256") for stage in stages)
            ):
                msg = "Target compiler stages are incomplete or out of order"
                raise ValueError(msg)
            if compilation.get("source_sha256") != source_hash or stages[0]["code"] != source["code"]:
                msg = "Target compilation uses a different application source"
                raise ValueError(msg)
            for stage in stages[:4]:
                shared = {key: stage.get(key) for key in ("code", "circuit")}
                if not shared["circuit"] or common_stages.setdefault(stage["id"], shared) != shared:
                    msg = "Target-independent stage source or circuit differs between targets"
                    raise ValueError(msg)
            if compilation.get("execution", {}).get("capture_script_sha256") != provenance["execution_script_sha256"]:
                msg = "Target execution provenance differs from its capture"
                raise ValueError(msg)
            metadata = target.get("provenance", {})
            raw = metadata.get("raw", {})
            # The IBM capture hashes its configuration/property pair in this recorded order.
            encoded = [raw["configuration"], raw["properties"]] if set(raw) == {"configuration", "properties"} else raw
            if (
                not raw
                or not metadata.get("source_url")
                or not metadata.get("retrieved_at")
                or hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest()
                != metadata.get("raw_sha256")
            ):
                msg = "Target metadata provenance differs from its original source"
                raise ValueError(msg)
    for target in data.get("targets", []):
        validate_capture({
            "schema_version": 1,
            "provenance": data["provenance"],
            "device": target["metadata"],
            "scenarios": [{"id": target["id"], "variants": [target["compilation"]]}],
        })
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
    formatter = HtmlFormatter(cssclass="highlight", nowrap=False, style="friendly")
    line_formatter = HtmlFormatter(nowrap=True, style="friendly")
    variants = [variant for scenario in data["scenarios"] for variant in scenario["variants"]]
    variants.extend(target["compilation"] for target in data.get("targets", []))
    for variant in variants:
        for artifact in [*variant.get("stages", []), *variant.get("exports", [])]:
            if "code" in artifact:
                code = artifact["code"]
                artifact["sha256"] = hashlib.sha256(code.encode()).hexdigest()
                lines = code.splitlines(keepends=True)
                artifact["line_count"] = len(lines)
                if len(lines) > 200:
                    artifact["full_code_gzip"] = base64.b64encode(gzip.compress(code.encode(), mtime=0)).decode("ascii")
                    focus_line = artifact.get("focus_line", 1)
                    if artifact.get("language") == "mlir":
                        focus_line = next(
                            (i + 1 for i, line in enumerate(lines) if re.search(r"\b(?:qc|qco)\.(?!static)", line)),
                            focus_line,
                        )
                    start = max(0, min(focus_line - 2, len(lines) - 200))
                    excerpt = lines[start : start + 200]
                    artifact["code"] = "".join(excerpt)
                    artifact["excerpt_lines"] = len(excerpt)
                    artifact["excerpt_start_line"] = start + 1
                language = artifact.get("language", "text")
                lexer = get_lexer_by_name("openqasm3" if language == "qasm" else language)
                artifact["html"] = highlight(artifact["code"], lexer, formatter)
                artifact["lines_html"] = [
                    highlight(line, lexer, line_formatter).rstrip("\n") for line in artifact["code"].splitlines()
                ]
    for variant in variants:
        for execution in [variant.get("execution"), *variant.get("executions", {}).values()]:
            if execution and execution.get("client_source"):
                execution["client_source_lines_html"] = [
                    highlight(line, get_lexer_by_name("python"), line_formatter).rstrip("\n")
                    for line in execution["client_source"].splitlines()
                ]
    application = data.get("application", {})
    for key in ("source", "batch_source", "classical_source"):
        if application.get(key):
            application[key + "_lines_html"] = [
                highlight(line, get_lexer_by_name("python"), line_formatter).rstrip("\n")
                for line in application[key].splitlines()
            ]
    if application.get("execution", {}).get("client_source"):
        execution = application["execution"]
        execution["client_source_lines_html"] = [
            highlight(line, get_lexer_by_name("python"), line_formatter).rstrip("\n")
            for line in execution["client_source"].splitlines()
        ]
    css = source.joinpath("presentation.css").read_text(encoding="utf-8")
    css = formatter.get_style_defs(".highlight") + "\n" + css
    script = source.joinpath("presentation.js").read_text(encoding="utf-8")
    script = (
        "\n".join(
            source.joinpath(name).read_text(encoding="utf-8")
            for name in ("visuals.js", "circuit.js", "motion.js")
            if source.joinpath(name).exists()
        )
        + "\n"
        + script
    )
    assets = {
        path.stem: "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()
        for path in sorted(source.glob("assets/*.svg"))
    }
    font = source / "assets/inter-latin.woff2"
    if font.exists():
        css = (
            "@font-face{font-family:Inter;src:url(data:font/woff2;base64,"
            + base64.b64encode(font.read_bytes()).decode()
            + ") format('woff2');font-weight:100 900;font-style:normal;font-display:block;}\n"
            + css
        )
    html = source.joinpath("index.html").read_text(encoding="utf-8")
    substitutions = {
        "<!-- MQSF_STYLES -->": f"<style>{css}</style>",
        "<!-- MQSF_DATA -->": (
            f"<script>window.MQSF_DATA={encode_data(data)};</script>"
            f"<script>window.MQSF_ASSETS={encode_data(assets)};</script>"
        ),
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
        archive.writestr(
            "evidence.json.gz", gzip.compress(gzip.decompress(raw) if capture.suffix == ".gz" else raw, mtime=0)
        )
        for asset in sorted(source.joinpath("assets").glob("*")):
            if asset.is_file():
                archive.write(asset, f"assets/{asset.name}")
        for name in ("presenter-notes.md", "README.md"):
            if source.joinpath(name).exists():
                archive.write(source / name, name)
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
