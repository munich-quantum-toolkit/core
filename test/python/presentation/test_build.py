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

import pytest

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
                "id": "test",
                "variants": [
                    {
                        "id": "preserved",
                        "stages": [{"id": "input", "language": "text", "code": code}],
                        "exports": [{"id": "qasm", "language": "text", "code": code}],
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


def test_embedded_source_cannot_end_script_and_roundtrips() -> None:
    """Keep source text intact without allowing script injection."""
    data = sample_capture()
    encoded = BUILDER.encode_data(data)
    assert "</script" not in encoded
    assert json.loads(encoded) == data


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
        source = root / "capture.json"
        data = sample_target_capture()
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
        assert packaged["target_provenance"] == data["target_provenance"]
        for target, original in zip(packaged["targets"], data["targets"], strict=True):
            assert target["provenance"] == original["provenance"]
            assert target["compilation"]["stages"][0]["code"] == data["application"]["representative_source"]["code"]
            assert target["compilation"]["stages"][0]["lines_html"]
        with zipfile.ZipFile(result.parent / "mqsf-2026.zip") as archive:
            assert archive.read("index.html") == result.read_bytes()
            assert json.loads(gzip.decompress(archive.read("evidence.json.gz"))) == data


def test_large_artifacts_keep_exact_source_in_compressed_download() -> None:
    """Bound highlighted excerpts while preserving the complete source."""
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
        assert variant["exports"][0]["sha256"] == variant["execution"]["payload_sha256"]
        assert artifact["sha256"] == hashlib.sha256(code.encode()).hexdigest()
        assert artifact["line_count"] == 201
        assert artifact["code"] == "line\n" * 200
        assert gzip.decompress(base64.b64decode(artifact["full_code_gzip"])).decode() == code
