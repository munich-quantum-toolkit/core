# Copyright (c) 2026 Munich Quantum Software Company GmbH
# SPDX-License-Identifier: MIT

# /// script
# dependencies = ["pygments>=2.19,<3", "mlir-pygments==1.0.0", "openqasm-pygments>=0.2,<0.3"]
# ///
"""Package recorded MQSF demonstrations without loading native MQT Core."""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path
from typing import Any, ClassVar

from capture_execution import validate_execution
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import inherit
from pygments.lexers import get_lexer_by_name
from pygments.lexers.asm import LlvmLexer
from pygments.token import Number

HERE = Path(__file__).resolve().parent


class QIRLexer(LlvmLexer):
    """Recognize LLVM 23's hexadecimal floating constants until Pygments does."""

    tokens: ClassVar = {"root": [(r"f0x[0-9A-Fa-f]+", Number.Float), inherit]}


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
    if evidence := data.get("compiler_evidence"):
        if evidence.get("provenance", {}).get("core_revision") != data["provenance"]["core_revision"]:
            msg = "Compiler evidence revision differs from the captured programs"
            raise ValueError(msg)
        for key in ("source", "before", "after", "implementation"):
            if (artifact := evidence.get("optimization", {}).get(key)) and artifact.get("sha256") != hashlib.sha256(
                artifact["code"].encode()
            ).hexdigest():
                msg = "Optimization artifact hash differs from its original source"
                raise ValueError(msg)
        if weyl := evidence.get("weyl"):
            target = next((target for target in data.get("targets", []) if target["id"] == weyl.get("target_id")), None)
            compilation = target["compilation"] if target else {}
            trace_hash = hashlib.sha256(json.dumps(compilation.get("synthesis_trace", []), sort_keys=True).encode())
            if (
                target is None
                or weyl.get("program_sha256") != compilation.get("source_sha256")
                or weyl.get("input_sha256") != trace_hash.hexdigest()
            ):
                msg = "Weyl evidence differs from its captured target program or fusion trace"
                raise ValueError(msg)
    if targets := data.get("targets"):
        application = data.get("application", {})
        source = application.get("representative_source", {})
        source_hash = hashlib.sha256(source.get("code", "").encode()).hexdigest()
        snapshots = application.get("snapshots", [])
        provenance = data.get("target_provenance", {})
        if (
            not source.get("code")
            or source.get("sha256") != source_hash
            or provenance.get("source_sha256") != source_hash
            or not snapshots
            or snapshots[0].get("input_payload", snapshots[0].get("payload")) != source["code"]
            or snapshots[0].get("input_payload_sha256", snapshots[0].get("payload_sha256")) != source_hash
        ):
            msg = "Target captures must identify the exact submitted application source"
            raise ValueError(msg)
        for snapshot in snapshots:
            payload = snapshot.get("payload", "")
            if hashlib.sha256(payload.encode()).hexdigest() != snapshot.get("payload_sha256"):
                msg = "Submitted application payload hash disagrees"
                raise ValueError(msg)
        compilations = application.get("execution", {}).get("compilation")
        if compilations is not None:
            if len(compilations) != len(snapshots) or {event["program_index"] for event in compilations} != set(
                range(len(snapshots))
            ):
                msg = "Application compilation indices differ from submitted programs"
                raise ValueError(msg)
            for event in compilations:
                snapshot = snapshots[event["program_index"]]
                if (
                    hashlib.sha256(snapshot["input_payload"].encode()).hexdigest() != snapshot["input_payload_sha256"]
                    or event["input_sha256"] != snapshot["input_payload_sha256"]
                    or event["output_sha256"] != snapshot["payload_sha256"]
                ):
                    msg = "Application compiler input/output binding differs"
                    raise ValueError(msg)
        # Execution and compilation may be recorded separately; their shared input is checked above.
        if not application.get("provenance", {}).get("core_revision") or any(
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
            qpy = next((artifact for artifact in variant.get("exports", []) if artifact["id"] == "qpy"), None)
            if qpy is not None:
                native = next(stage for stage in variant["stages"] if stage["id"] == "target-native-synthesis")
                if (
                    qpy["provenance"]["native_ir_sha256"] != hashlib.sha256(native["code"].encode()).hexdigest()
                    or qpy["qpy_sha256"]
                    != hashlib.sha256(base64.b64decode(qpy["qpy_base64"], validate=True)).hexdigest()
                    or qpy["svg_sha256"] != hashlib.sha256(qpy["svg"].encode()).hexdigest()
                ):
                    msg = "QPY export or drawing differs from its captured source"
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


def browser_data(data: dict[str, Any]) -> dict[str, Any]:
    """Select the recorded fields consumed by the deck; retain full evidence in the ZIP.

    Returns:
        The explicit browser projection, sharing unchanged records with the capture.
    """
    scenarios = []
    for scenario in data["scenarios"]:
        if scenario["id"] not in {"parity", "rus", "qpe"}:
            continue
        variant = next(variant for variant in scenario["variants"] if variant["id"] == "structured")
        shown = {
            "id": variant["id"],
            "stages": [stage for stage in variant["stages"] if stage["id"] == "source"],
            "exports": [
                artifact
                for artifact in variant["exports"]
                if scenario["id"] != "qpe" and artifact["id"] == "qir-adaptive"
            ],
        }
        if scenario["id"] == "qpe":
            shown["executions"] = {"qir-adaptive": variant["executions"]["qir-adaptive"]}
        scenarios.append({"id": scenario["id"], "variants": [shown]})
    targets = []
    for index, target in enumerate(data.get("targets", [])):
        compilation = target["compilation"]
        stages = []
        for stage in compilation["stages"]:
            if stage["id"] not in {"source", "qc", "qco", "optimized", "place-and-route", "target-native-synthesis"}:
                continue
            # Only the first target's input IR is displayed as source; all target circuits are replayed.
            stages.append(
                stage
                if index == 0 and stage["id"] in {"source", "qc", "qco"}
                else {key: value for key, value in stage.items() if key in {"id", "label", "circuit"}}
            )
        targets.append({
            **{key: target[key] for key in ("id", "label", "metadata", "compiler_model") if key in target},
            "compilation": {
                **{
                    key: compilation[key]
                    for key in ("layout", "routing_trace", "metrics", "synthesis_trace")
                    if key in compilation
                },
                "stages": stages,
                "exports": [
                    artifact
                    for artifact in compilation["exports"]
                    if artifact["id"]
                    == {"ibm.ibm_miami": "qpy", "aws.ionq.forte-1": "openqasm3"}.get(target["id"], "qir-adaptive")
                ],
            },
        })
    application = data.get("application", {})
    propagation = application.get("propagation", {})
    curves = []
    for curve in propagation.get("curves", []):
        shown = {key: curve[key] for key in ("label", "energy", "stderr", "step_completed_ms") if key in curve}
        if "frames" in curve:
            shown["frames"] = [
                {
                    "tau": frame["tau"],
                    "step": frame["step"],
                    "walkers": [{key: w[key] for key in ("id", "weight", "completed_ms")} for w in frame["walkers"]],
                }
                for frame in curve["frames"]
            ]
        curves.append(shown)
    return {
        **{key: data[key] for key in ("schema_version", "provenance", "device", "references") if key in data},
        "scenarios": scenarios,
        "targets": targets,
        "compiler_evidence": {
            key: value
            for key, value in data.get("compiler_evidence", {}).items()
            if key in {"grover_size", "optimization", "weyl"}
        },
        "application": {
            **{
                key: application[key]
                for key in (
                    "source",
                    "batch_source",
                    "source_definitions",
                    "presentation_source",
                    "circuit_sections",
                    "workload",
                    "chemistry",
                    "execution",
                    "slurm",
                    "timeline",
                )
                if key in application
            },
            "shadow_preview": [
                {"program_index": snapshot["native_program_index"], "shots": snapshot["shots"][:3]}
                for snapshot in application.get("snapshots", [])[:4]
            ],
            "propagation": {
                **{
                    key: propagation[key]
                    for key in ("tau", "walkers", "workers", "duration_ms", "wall_time_ms")
                    if key in propagation
                },
                "curves": curves,
            },
        },
    }


def build(output: Path, capture: Path = HERE / "captures/demo.json.gz", source: Path = HERE) -> Path:
    """Build standalone HTML and separate presentation and evidence archives.

    Returns:
        Path to the self-contained HTML presentation.

    Raises:
        ValueError: If captures or template markers are inconsistent.
    """
    raw = capture.read_bytes()
    data = json.loads(gzip.decompress(raw) if capture.suffix == ".gz" else raw)
    validate_capture(data)
    data = browser_data(data)
    formatter = HtmlFormatter(cssclass="highlight", nowrap=False, style="friendly")
    line_formatter = HtmlFormatter(nowrap=True, style="friendly")
    variants = [variant for scenario in data["scenarios"] for variant in scenario["variants"]]
    variants.extend(target["compilation"] for target in data.get("targets", []))
    optimization = data.get("compiler_evidence", {}).get("optimization", {})
    variants.append({"stages": [optimization[key] for key in ("before", "after") if key in optimization]})
    for variant in variants:
        for artifact in [*variant.get("stages", []), *variant.get("exports", [])]:
            if "code" in artifact:
                code = artifact["code"]
                if "sha256" not in artifact:
                    artifact["sha256"] = hashlib.sha256(code.encode()).hexdigest()
                lines = code.splitlines(keepends=True)
                artifact["line_count"] = len(lines)
                language = artifact.get("language", "text")
                lexer = (
                    QIRLexer()
                    if language == "llvm"
                    else get_lexer_by_name("openqasm3" if language == "qasm" else language)
                )
                if artifact.get("display_code"):
                    artifact["display_lines_html"] = [
                        highlight(line, lexer, line_formatter).rstrip("\n")
                        for line in artifact["display_code"].splitlines()
                    ]
                artifact["lines_html"] = [
                    highlight(line, lexer, line_formatter).rstrip("\n") for line in artifact["code"].splitlines()
                ]
    application = data.get("application", {})
    definitions = application.get("source_definitions", {})
    if definitions:

        def executable_lines(code: str) -> str:
            code = textwrap.dedent(code)
            lines = code.splitlines()
            omitted = set()
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
                    docstring = node.body[0]
                    omitted.update(range(docstring.lineno - 1, docstring.end_lineno))
            return "\n".join(line for i, line in enumerate(lines) if i not in omitted and line.strip())

        application["presentation_source"] = {
            "left": "\n\n".join(
                executable_lines(definitions[name]) for name in ("hydrogen_trial", "apply_gaussian_givens")
            ),
            "right": "\n\n".join((
                executable_lines(definitions["apply_pauli_layer"]),
                textwrap.dedent(application["source"]).strip(),
                application["batch_source"],
            )),
            "overview": textwrap.dedent(application["source"]).strip() + "\n\n" + application["batch_source"],
        }
        for key, code in list(application["presentation_source"].items()):
            application["presentation_source"][key + "_lines_html"] = [
                highlight(line, get_lexer_by_name("python"), line_formatter).rstrip("\n") for line in code.splitlines()
            ]
    for key in ("source", "batch_source"):
        if application.get(key):
            application[key + "_lines_html"] = [
                highlight(line, get_lexer_by_name("python"), line_formatter).rstrip("\n")
                for line in application[key].splitlines()
            ]
    css = source.joinpath("presentation.css").read_text(encoding="utf-8")
    if source.joinpath("opening.css").exists():
        css += "\n" + source.joinpath("opening.css").read_text(encoding="utf-8")
    css = formatter.get_style_defs(".highlight") + "\n" + css
    script = source.joinpath("presentation.js").read_text(encoding="utf-8")
    script = (
        "\n".join(
            source.joinpath(name).read_text(encoding="utf-8")
            for name in ("visuals.js", "circuit.js", "motion.js", "opening.js")
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
    notices = "\n\n".join(
        f"{path.name}\n{'=' * len(path.name)}\n\n{path.read_text(encoding='utf-8')}"
        for path in [
            HERE.parents[1] / "LICENSE",
            *sorted(source.glob("assets/*-LICENSE.txt")),
            source / "assets/sources.json",
        ]
        if path.is_file()
    )
    with zipfile.ZipFile(output / "mqsf-2026.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result, "index.html")
        archive.writestr("NOTICES.txt", notices)
        for name, packaged_name in (("offline-readme.txt", "README.txt"), ("presenter-notes.md", "presenter-notes.md")):
            if source.joinpath(name).exists():
                archive.write(source / name, packaged_name)
    with zipfile.ZipFile(output / "mqsf-2026-evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("NOTICES.txt", notices)
        archive.writestr("evidence.json.gz", raw if capture.suffix == ".gz" else gzip.compress(raw, mtime=0))
        for asset in sorted(source.joinpath("assets").glob("*")):
            if asset.is_file():
                archive.write(asset, f"assets/{asset.name}")
        for name in ("README.md", "compiler-fixes.md"):
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
