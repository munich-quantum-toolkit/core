# Copyright (c) 2026 Munich Quantum Software Company GmbH
# SPDX-License-Identifier: MIT

"""Check the offline bundle boundary without importing native MQT Core."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pygments.token import Error, Number

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("mqsf_build", ROOT / "presentations/mqsf2026/build.py")
assert SPEC is not None
assert SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(ROOT / "presentations/mqsf2026"))
    SPEC.loader.exec_module(BUILDER)


def sample_capture() -> dict:
    """Return small test data, never distributed as presentation evidence."""
    code = 'OPENQASM 3.0; // </script><script>alert("test")</script>\n'
    return {
        "schema_version": 1,
        "title": "Test presentation",
        "provenance": {"core_revision": "a" * 40},
        "device": {"sites": [{"id": 0, "name": "QB1"}], "edges": []},
        "scenarios": [
            {
                "id": "parity",
                "variants": [
                    {
                        "id": "structured",
                        "stages": [{"id": "source", "language": "text", "code": code}],
                        "exports": [{"id": "qir-adaptive", "language": "text", "code": code}],
                        "layout": {"initial": [0], "final": [0], "swaps": []},
                        "execution": {
                            "payload_sha256": hashlib.sha256(code.encode()).hexdigest(),
                            "shots": ["0", "1", "0"],
                            "counts": {"0": 2, "1": 1},
                            "terminal_status": "DONE",
                            "payload_identity_verified": True,
                            "num_shots": 3,
                            "events": [{"time_ms": 0}, {"time_ms": 1}],
                        },
                    }
                ],
            }
        ],
    }


def sample_target_capture() -> dict:
    """Return two tiny consistent target captures for merge-boundary regression tests."""
    data = sample_capture()
    code = data["scenarios"][0]["variants"][0]["stages"][0]["code"]
    source_hash = hashlib.sha256(code.encode()).hexdigest()
    data["application"] = {
        "representative_source": {"code": code, "sha256": source_hash},
        "snapshots": [{"payload": code, "payload_sha256": source_hash, "native_program_index": 0, "shots": ["0"]}],
        "provenance": {"core_revision": data["provenance"]["core_revision"]},
    }
    data["provenance"].update({"compiler_sha256": "c" * 64, "execution_script_sha256": "d" * 64})
    data["target_provenance"] = {
        "source_sha256": source_hash,
        "compiler_sha256": "c" * 64,
        "execution_script_sha256": "d" * 64,
    }
    data["targets"] = []
    for identifier in ("target-a", "target-b"):
        compilation = deepcopy(data["scenarios"][0]["variants"][0])
        compilation["source_sha256"] = source_hash
        compilation["execution"]["capture_script_sha256"] = "d" * 64
        compilation["stages"] = [
            {
                "id": stage,
                "code": code,
                "sha256": source_hash,
                "language": "text",
                "circuit": {"qubits": [{"id": 0}], "operations": []},
            }
            for stage in ("source", "qc", "qco", "optimized", "place-and-route", "target-native-synthesis")
        ]
        raw = {"device": identifier}
        data["targets"].append({
            "id": identifier,
            "metadata": deepcopy(data["device"]),
            "compilation": compilation,
            "provenance": {
                "source_url": "https://example.com/device",
                "retrieved_at": "2026-10-10T00:00:00Z",
                "raw": raw,
                "raw_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
            },
        })
    return data


@pytest.mark.parametrize(
    "failure",
    [
        "application-source",
        "application-provenance",
        "submitted-source",
        "target-source",
        "target-digest",
        "stage-source",
        "stage-circuit",
        "stage-order",
        "metadata",
        "execution",
        "merge",
        "duplicate",
    ],
)
def test_rejects_inconsistent_application_target_merge(failure: str) -> None:
    """Bind every target, compiler view, and provenance record to the actual application."""
    data = sample_target_capture()
    BUILDER.validate_capture(data)
    target = data["targets"][1]
    compilation = target["compilation"]
    if failure == "application-source":
        data["application"]["representative_source"]["code"] += "changed"
    elif failure == "application-provenance":
        del data["application"]["provenance"]["core_revision"]
    elif failure == "submitted-source":
        snapshot = data["application"]["snapshots"][0]
        snapshot["payload"] += "changed"
        snapshot["payload_sha256"] = hashlib.sha256(snapshot["payload"].encode()).hexdigest()
    elif failure == "target-source":
        stage = compilation["stages"][0]
        stage["code"] += "changed"
        stage["sha256"] = hashlib.sha256(stage["code"].encode()).hexdigest()
    elif failure == "target-digest":
        compilation["source_sha256"] = "b" * 64
    elif failure == "stage-source":
        stage = compilation["stages"][2]
        stage["code"] += "changed"
        stage["sha256"] = hashlib.sha256(stage["code"].encode()).hexdigest()
    elif failure == "stage-circuit":
        compilation["stages"][2]["circuit"]["operations"].append({"name": "x", "qubits": [0]})
    elif failure == "stage-order":
        compilation["stages"][-2:] = compilation["stages"][:-3:-1]
    elif failure == "metadata":
        target["provenance"]["raw"]["device"] = "a different device"
    elif failure == "execution":
        compilation["execution"]["capture_script_sha256"] = "b" * 64
    elif failure == "merge":
        data["target_provenance"]["compiler_sha256"] = "b" * 64
    else:
        target["id"] = data["targets"][0]["id"]
    with pytest.raises(ValueError, match=r"source|stages|provenance|Duplicate target"):
        BUILDER.validate_capture(data)


@pytest.mark.parametrize("failure", ["source", "input-hash", "output-hash", "index", "duplicate", "missing"])
def test_application_compilation_binds_each_input_to_its_executable(failure: str) -> None:
    """Reject a stale or incomplete QASM-to-QIR batch before packaging its replay."""
    data = sample_target_capture()
    application = data["application"]
    snapshot = application["snapshots"][0]
    snapshot["input_payload"] = snapshot["payload"]
    snapshot["input_payload_sha256"] = snapshot["payload_sha256"]
    snapshot["payload"] = "; distinct executable QIR payload\n"
    snapshot["payload_sha256"] = hashlib.sha256(snapshot["payload"].encode()).hexdigest()
    application["snapshots"].append(deepcopy(snapshot))
    events = [
        {
            "program_index": i,
            "input_sha256": snapshot["input_payload_sha256"],
            "output_sha256": snapshot["payload_sha256"],
        }
        for i in range(2)
    ]
    application["execution"] = {"compilation": events}
    BUILDER.validate_capture(data)
    if failure == "source":
        application["snapshots"][1]["input_payload"] += "changed"
    elif failure == "input-hash":
        events[1]["input_sha256"] = "b" * 64
    elif failure == "output-hash":
        events[1]["output_sha256"] = "b" * 64
    elif failure == "index":
        events[1]["program_index"] = -1
    elif failure == "duplicate":
        events[1]["program_index"] = 0
    else:
        events.clear()
    with pytest.raises(ValueError, match="Application compil"):
        BUILDER.validate_capture(data)


def test_application_recording_can_precede_compiler_capture() -> None:
    """Compile the same retained payload with a newer compiler without rewriting its history."""
    data = sample_target_capture()
    data["application"]["provenance"]["core_revision"] = "b" * 40
    original = deepcopy(data)
    BUILDER.validate_capture(data)
    assert data == original


def test_embedded_source_cannot_end_script_and_roundtrips() -> None:
    """Keep source text intact without allowing script injection."""
    data = sample_capture()
    encoded = BUILDER.encode_data(data)
    assert "</script" not in encoded
    assert json.loads(encoded) == data


@pytest.mark.parametrize("failure", ["revision", "optimization", "weyl-target", "weyl-program", "weyl-trace"])
def test_rejects_compiler_evidence_from_different_capture(failure: str) -> None:
    """Bind displayed optimization and Weyl evidence before creating browser excerpts."""
    data = sample_target_capture()
    target = data["targets"][0]
    target["compilation"]["synthesis_trace"] = [{"matrix": "recorded fusion"}]
    data["compiler_evidence"] = {
        "provenance": {"core_revision": data["provenance"]["core_revision"]},
        "optimization": {"after": deepcopy(target["compilation"]["stages"][0])},
        "weyl": {
            "target_id": target["id"],
            "program_sha256": target["compilation"]["source_sha256"],
            "input_sha256": hashlib.sha256(
                json.dumps(target["compilation"]["synthesis_trace"], sort_keys=True).encode()
            ).hexdigest(),
        },
    }
    BUILDER.validate_capture(data)
    evidence: dict[str, Any] = data["compiler_evidence"]
    if failure == "revision":
        evidence["provenance"]["core_revision"] = "b" * 40
    elif failure == "optimization":
        evidence["optimization"]["after"]["code"] += "changed"
    elif failure == "weyl-target":
        evidence["weyl"]["target_id"] = "missing-target"
    elif failure == "weyl-program":
        evidence["weyl"]["program_sha256"] = "b" * 64
    else:
        target["compilation"]["synthesis_trace"].append({"matrix": "another fusion"})
    with pytest.raises(ValueError, match=r"Compiler evidence|Optimization artifact|Weyl evidence"):
        BUILDER.validate_capture(data)


def test_rejects_histogram_that_does_not_match_shots() -> None:
    """Reject contradictory execution evidence."""
    data = sample_capture()
    data["scenarios"][0]["variants"][0]["execution"]["counts"]["0"] = 1
    with pytest.raises(ValueError, match="histogram"):
        BUILDER.validate_capture(data)


def test_rejects_payload_that_is_not_a_captured_export() -> None:
    """Reject a result from a different payload."""
    data = sample_capture()
    data["scenarios"][0]["variants"][0]["execution"]["payload_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="payload"):
        BUILDER.validate_capture(data)


def test_rejects_nonexistent_topology_endpoint() -> None:
    """Reject edges outside the captured device."""
    data = sample_capture()
    data["device"]["edges"] = [[0, 54]]
    with pytest.raises(ValueError, match="topology"):
        BUILDER.validate_capture(data)


def test_rejects_incorrect_artifact_hash() -> None:
    """Do not present a digest belonging to different compiler output."""
    data = sample_capture()
    data["scenarios"][0]["variants"][0]["exports"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Artifact hash"):
        BUILDER.validate_capture(data)


@pytest.mark.parametrize("changed", ["native_ir_sha256", "qpy_base64", "svg"])
def test_qpy_drawing_and_serialization_are_bound_to_native_ir(changed: str) -> None:
    """Reject stale drawings or binary payloads while projecting each target's own format."""
    data = sample_target_capture()
    ibm, ionq = data["targets"]
    ibm["id"], ionq["id"] = "ibm.ibm_miami", "aws.ionq.forte-1"
    artifact = {
        "id": "qpy",
        "code": "qpy.dump(circuit, file)",
        "sha256": hashlib.sha256(b"qpy.dump(circuit, file)").hexdigest(),
        "qpy_base64": base64.b64encode(b"qpy-test").decode(),
        "qpy_sha256": hashlib.sha256(b"qpy-test").hexdigest(),
        "svg": "<svg/>",
        "svg_sha256": hashlib.sha256(b"<svg/>").hexdigest(),
        "provenance": {"native_ir_sha256": ibm["compilation"]["stages"][-1]["sha256"]},
    }
    ibm["compilation"]["exports"].append(artifact)
    ionq["compilation"]["exports"].append({"id": "openqasm3", "code": "native IonQ"})
    BUILDER.validate_capture(data)
    projected = BUILDER.browser_data(data)["targets"]
    assert projected[0]["compilation"]["exports"] == [artifact]
    assert projected[1]["compilation"]["exports"] == [{"id": "openqasm3", "code": "native IonQ"}]
    if changed == "native_ir_sha256":
        artifact["provenance"][changed] = "0" * 64
    elif changed == "qpy_base64":
        artifact[changed] = base64.b64encode(b"changed").decode()
    else:
        artifact["svg"] = "<svg>changed</svg>"
    with pytest.raises(ValueError, match="QPY export or drawing"):
        BUILDER.validate_capture(data)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("terminal_status", "FAILED"),
        ("payload_identity_verified", False),
        ("num_shots", 999),
        ("events", [{"time_ms": 1}, {"time_ms": 0}]),
    ],
)
def test_rejects_invalid_execution_metadata(field: str, invalid: object) -> None:
    """Reject failed, changed, incomplete, or reordered execution records."""
    data = sample_capture()
    data["scenarios"][0]["variants"][0]["execution"][field] = invalid
    with pytest.raises(ValueError, match=r"Execution must|Returned shot count|capture order"):
        BUILDER.validate_capture(data)


def test_rejects_invalid_alternate_format_execution() -> None:
    """Check selectable alternate executions, not just the default capture."""
    data = sample_capture()
    variant = data["scenarios"][0]["variants"][0]
    variant["executions"] = {"qasm": variant["execution"] | {"terminal_status": "FAILED"}}
    with pytest.raises(ValueError, match="Execution must"):
        BUILDER.validate_capture(data)


def test_rejects_execution_key_that_names_a_different_export() -> None:
    """Bind each execution to its own exported program."""
    data = sample_capture()
    variant = data["scenarios"][0]["variants"][0]
    variant["exports"].append({"id": "other", "code": "different payload", "language": "text"})
    variant["executions"] = {"other": variant["execution"]}
    with pytest.raises(ValueError, match="payload"):
        BUILDER.validate_capture(data)


def test_bundle_is_self_contained_and_keeps_exact_source() -> None:
    """Package all runtime content into one HTML file."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "index.html").write_text(
            "<html><head><!-- MQSF_STYLES --></head><body><!-- MQSF_DATA --><!-- MQSF_SCRIPT --></body></html>",
            encoding="utf-8",
        )
        (root / "presentation.css").write_text("body { color: white; }", encoding="utf-8")
        (root / "presentation.js").write_text("window.ready = true;", encoding="utf-8")
        for name in ("visuals", "circuit", "motion"):
            (root / f"{name}.js").write_text(f"window.{name} = true;", encoding="utf-8")
        (root / "assets").mkdir()
        (root / "assets/equation.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        (root / "assets/inter-latin.woff2").write_bytes(b"font test bytes")
        (root / "assets/sources.json").write_text('{"logo": "upstream attribution"}', encoding="utf-8")
        (root / "assets/logo-LICENSE.txt").write_text("Upstream license", encoding="utf-8")
        (root / "offline-readme.txt").write_text("Open index.html", encoding="utf-8")
        (root / "presenter-notes.md").write_text("# Speaker notes", encoding="utf-8")
        (root / "README.md").write_text("# Evidence and development", encoding="utf-8")
        (root / "compiler-fixes.md").write_text("# Compiler inventory", encoding="utf-8")
        source = root / "capture.json"
        data = sample_target_capture()
        data["application"].update({
            "source": "    @decorator\n    def shadow_circuit():\n        return 1\n",
            "batch_source": "compiled = [1]\nschedule = compiled[0]",
            "source_definitions": {
                name: f'def {name}():\n    """Description only."""\n    return 1\n'
                for name in ("hydrogen_trial", "apply_gaussian_givens", "apply_pauli_layer")
            },
            "workload": {"snapshots": 1},
            "shadow": {"raw": "scientific evidence"},
            "propagation": {
                "curves": [
                    {
                        "energy": [1],
                        "walker_energies": [[1]],
                        "walker_weights": [[1]],
                        "step_completed_ms": [3],
                        "frames": [
                            {"tau": 0, "step": 0, "walkers": [{"id": 0, "weight": 1, "energy": 2, "completed_ms": 4}]}
                        ],
                    },
                    {
                        "energy": [2],
                        "frames": [{"tau": 0, "step": 0, "walkers": [{"id": 0, "weight": 2, "completed_ms": 5}]}],
                    },
                ]
            },
        })
        data["targets"][0]["compilation"]["stages"].append({
            "id": "unused-intermediate",
            "language": "text",
            "code": "retained only in evidence",
            "sha256": hashlib.sha256(b"retained only in evidence").hexdigest(),
        })
        source.write_text(json.dumps(data), encoding="utf-8")
        result = BUILDER.build(root / "output", source, root)
        html = result.read_text(encoding="utf-8")
        assert "<!-- MQSF_" not in html
        assert "<script src=" not in html
        assert "window.MQSF_DATA=" in html
        assert "window.ready = true;" in html
        for name in ("visuals", "circuit", "motion"):
            assert html.index(f"window.{name} = true;") < html.index("window.ready = true;")
        assert "data:image/svg+xml;base64," in html
        assert "data:font/woff2;base64," in html
        packaged = json.loads(html.split("window.MQSF_DATA=", 1)[1].split(";</script>", 1)[0])
        display = packaged["application"]["presentation_source"]
        assert "\n@decorator\ndef shadow_circuit():" in display["right"]
        assert "Description only" not in display["left"]
        assert "<span" in "".join(display["right_lines_html"])
        assert "snapshots" not in packaged["application"]
        assert "shadow" not in packaged["application"]
        assert packaged["application"]["workload"]["snapshots"] == 1
        assert packaged["application"]["propagation"]["curves"] == [
            {
                "energy": [1],
                "step_completed_ms": [3],
                "frames": [{"tau": 0, "step": 0, "walkers": [{"id": 0, "weight": 1, "completed_ms": 4}]}],
            },
            {"energy": [2], "frames": [{"tau": 0, "step": 0, "walkers": [{"id": 0, "weight": 2, "completed_ms": 5}]}]},
        ]
        first, second = packaged["targets"]
        assert first["compilation"]["stages"][0]["code"] == data["application"]["representative_source"]["code"]
        assert first["compilation"]["stages"][0]["lines_html"]
        assert all(stage["id"] != "unused-intermediate" for stage in first["compilation"]["stages"])
        assert "code" not in second["compilation"]["stages"][0]
        for target, original in zip(packaged["targets"], data["targets"], strict=True):
            assert "provenance" not in target
            assert target["compilation"]["stages"][0]["circuit"] == original["compilation"]["stages"][0]["circuit"]
        with zipfile.ZipFile(result.parent / "mqsf-2026.zip") as archive:
            assert set(archive.namelist()) == {"index.html", "README.txt", "presenter-notes.md", "NOTICES.txt"}
            assert archive.read("index.html") == result.read_bytes()
            assert archive.read("README.txt") == (root / "offline-readme.txt").read_bytes()
            assert archive.read("presenter-notes.md") == (root / "presenter-notes.md").read_bytes()
            notices = archive.read("NOTICES.txt").decode()
            assert (ROOT / "LICENSE").read_text() in notices
            assert "Upstream license" in notices
            assert "upstream attribution" in notices
        with zipfile.ZipFile(result.parent / "mqsf-2026-evidence.zip") as archive:
            assert "index.html" not in archive.namelist()
            for asset in (root / "assets").iterdir():
                assert archive.read(f"assets/{asset.name}") == asset.read_bytes()
            for name in ("README.md", "compiler-fixes.md"):
                assert archive.read(name) == (root / name).read_bytes()
            assert json.loads(gzip.decompress(archive.read("evidence.json.gz"))) == data
            assert gzip.decompress(archive.read("evidence.json.gz")) == source.read_bytes()


def test_large_artifacts_keep_exact_source_in_evidence() -> None:
    """Keep full highlighted programs and identical downloadable evidence."""
    data = sample_capture()
    code = "line\n" * 201
    data["scenarios"][0]["variants"][0]["stages"][0]["code"] = code
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "index.html").write_text("<!-- MQSF_STYLES --><!-- MQSF_DATA --><!-- MQSF_SCRIPT -->", encoding="utf-8")
        (root / "presentation.css").write_text("", encoding="utf-8")
        (root / "presentation.js").write_text("", encoding="utf-8")
        capture = root / "capture.json.gz"
        capture.write_bytes(gzip.compress(json.dumps(data).encode()))
        html = BUILDER.build(root / "output", capture, root).read_text(encoding="utf-8")
        packaged = json.loads(html.split("window.MQSF_DATA=", 1)[1].split(";</script>", 1)[0])
        artifact = packaged["scenarios"][0]["variants"][0]["stages"][0]
        variant = packaged["scenarios"][0]["variants"][0]
        assert variant["exports"][0]["sha256"] == data["scenarios"][0]["variants"][0]["execution"]["payload_sha256"]
        assert artifact["sha256"] == hashlib.sha256(code.encode()).hexdigest()
        assert artifact["line_count"] == 201
        assert artifact["code"] == code
        assert len(artifact["lines_html"]) == 201
        assert "full_code_gzip" not in artifact
        with zipfile.ZipFile(root / "output/mqsf-2026-evidence.zip") as archive:
            assert archive.read("evidence.json.gz") == capture.read_bytes()
            assert json.loads(gzip.decompress(archive.read("evidence.json.gz"))) == data


def test_qir_hex_float_highlighting_keeps_actual_llvm_spelling() -> None:
    """LLVM 23 hex floats are numbers, not lexer errors or rewritten code."""
    code = "call void @gate(double f0xBFF921FB54442D18)\n"
    tokens = list(BUILDER.QIRLexer().get_tokens(code))
    assert "".join(value for _, value in tokens) == code
    assert not any(kind is Error for kind, _ in tokens)
    assert (Number.Float, "f0xBFF921FB54442D18") in tokens
