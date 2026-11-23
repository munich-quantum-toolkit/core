# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the QDMI Python bindings."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from collections import Counter
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from mqt.core.mlir import (
    OutputFormat,
    compile_program,
    submit_program,
)
from mqt.core.qdmi import (
    CustomProperty,
    Device,
    Job,
    ProgramFormat,
    Session,
    device_ids,
    is_binary_program_format,
    open_device,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_device_ids() -> None:
    """List the default session's client-visible stable IDs."""
    assert device_ids() == Session().device_ids


@pytest.mark.parametrize("parameter", [2**31, -(2**31) - 1])
def test_custom_integer_overflow_preserves_type(ddsim_device: Device, parameter: int) -> None:
    """Reject oversized integers instead of reinterpreting them as double bytes."""
    source = "OPENQASM 3.0; qubit q;"
    for submit in (ddsim_device.submit_job, ddsim_device.try_submit_job):
        with pytest.raises(TypeError, match="incompatible"):
            submit(source, ProgramFormat.QASM3, 1, custom3=parameter)
    with pytest.raises(TypeError, match="incompatible"):
        submit_program(source, target=ddsim_device, custom3=parameter)


@pytest.fixture
def ddsim_device() -> Device:
    """Return a local simulator session sized for these binding tests."""
    return open_device("mqt.ddsim.default", custom1="2")


@pytest.mark.parametrize("device_id", ["mqt.ddsim.default", "mqt.sc.default"])
def test_device_catalog(device_id: str) -> None:
    """Read local device metadata and retain its site and operation handles."""
    device = open_device(device_id, custom1="2") if device_id == "mqt.ddsim.default" else open_device(device_id)
    assert device.id == device_id
    assert device.name()
    assert device.version()
    assert device.library_version()
    assert isinstance(device.status(), Device.Status)
    sites = device.sites()
    assert len(sites) == device.qubits_num()
    assert sites[0].index() == 0
    assert not device.child_devices()
    operations = device.operations()
    assert operations
    assert all(operation.name() for operation in operations)
    for operation in operations:
        operation_sites = operation.sites()
        if operation_sites is not None:
            assert all(site in sites for site in operation_sites)
    assert sites[0].query_custom_property(CustomProperty.CUSTOM1, bytes) is None
    assert operations[0].query_custom_property(CustomProperty.CUSTOM1, bytes) is None


def test_custom_property_types(ddsim_device: Device) -> None:
    """Map absent optional values to None and reject unsupported Python types."""
    for value_type in (str, bool, int, float, bytes):
        assert ddsim_device.query_custom_property(CustomProperty.CUSTOM5, value_type) is None
    assert ddsim_device.query_custom_operations(CustomProperty.CUSTOM5) is None
    with pytest.raises(TypeError, match="value_type must be exactly"):
        ddsim_device.query_custom_property(CustomProperty.CUSTOM1, cast("type[bytes]", list))


def test_job_lifecycle(ddsim_device: Device) -> None:
    """Submit a text payload, retain metadata, and decode stable shots and counts."""
    source = 'OPENQASM 3; include "stdgates.inc"; qubit[2] q; x q[1]; bit[2] c = measure q;'
    job = ddsim_device.submit_job(source, ProgramFormat.QASM3, num_shots=4)
    assert job.id
    assert job.program_format == ProgramFormat.QASM3
    assert job.get_program() == source
    assert job.get_program(bytes) == source.encode() + b"\0"
    assert job.num_shots == 4
    assert job.queue_position is None
    assert isinstance(job.check(), Job.Status)
    assert job.query_custom_property(CustomProperty.CUSTOM1, bytes) is None
    assert job.wait()
    assert job.check() == Job.Status.DONE
    assert job.queue_position is None
    shots = job.get_shots()
    assert shots == ["10"] * 4
    assert Counter(shots) == job.get_counts()
    assert job.get_shots() == shots
    assert job.get_custom_result(CustomProperty.CUSTOM1, bytes) is None


def test_is_binary_program_format() -> None:
    """Classify the program formats that require exact-byte submission."""
    binary = {ProgramFormat.QIR_BASE_MODULE, ProgramFormat.QIR_ADAPTIVE_MODULE, ProgramFormat.QPY}
    for fmt in ProgramFormat:
        assert is_binary_program_format(fmt) == (fmt in binary), fmt.name


@pytest.mark.parametrize("program", [b"OPENQASM 3.0;", b"OPENQASM 3.0;\0garbage\0", "OPENQASM 3.0;\0garbage"])
def test_device_rejects_invalid_text_payloads(ddsim_device: Device, program: str | bytes) -> None:
    """Reject payloads that do not satisfy QDMI's text contract."""
    with pytest.raises(ValueError, match=r"(?:Setting programs: Invalid argument|embedded null bytes)"):
        ddsim_device.submit_job(program, ProgramFormat.QASM3, num_shots=1)


def test_device_rejects_text_for_binary_format(ddsim_device: Device) -> None:
    """Require exact byte submission for known binary formats."""
    with pytest.raises(ValueError, match="require exact-byte submission"):
        ddsim_device.submit_job("not bitcode", ProgramFormat.QIR_BASE_MODULE, num_shots=1)


def test_device_executes_qir_program(ddsim_device: Device) -> None:
    """Compile for and execute a QIR program with the DDSIM device."""
    qasm3_program = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""
    job = submit_program(
        qasm3_program, target=ddsim_device, program_format=ProgramFormat.QIR_BASE_STRING, num_shots=32, custom1=17
    )
    assert job.program_format == ProgramFormat.QIR_BASE_STRING
    job.wait()

    assert job.check() == Job.Status.DONE
    counts = job.get_counts()
    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == 32


def test_device_executes_binary_qir_program(ddsim_device: Device) -> None:
    """Submit and retrieve an exact QIR module byte payload."""
    qasm3_program = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""
    program = compile_program(qasm3_program, output=OutputFormat.QIR_BASE)
    program_bytes = program.to_bitcode()
    assert ProgramFormat.QIR_BASE_MODULE in ddsim_device.supported_program_formats()

    job = ddsim_device.submit_job(program_bytes, ProgramFormat.QIR_BASE_MODULE, num_shots=10)
    assert job.get_program(bytes) == program_bytes
    with pytest.raises(ValueError, match="binary program"):
        _ = job.get_program()
    job.wait()

    assert job.check() == Job.Status.DONE
    assert sum(job.get_counts().values()) == 10


def test_device_submit_job_handles_custom_parameters(ddsim_device: Device) -> None:
    """Test that submit_job forwards custom job parameters to DDSIM."""
    job = ddsim_device.submit_job(
        "OPENQASM 3.0; qubit q; bit c = measure q;", ProgramFormat.QASM3, 1, custom1=7, custom3=2
    )
    job.wait()
    assert job.check() == Job.Status.DONE

    with pytest.raises(ValueError, match=r"Setting custom parameter: Invalid argument\."):
        ddsim_device.submit_job("OPENQASM 3.0;", ProgramFormat.QASM3, 1, custom1="value")
    with pytest.raises(ValueError, match=r"Setting custom parameter: Invalid argument\."):
        ddsim_device.submit_job("OPENQASM 3.0;", ProgramFormat.QASM3, 1, custom2="value")
    with pytest.raises(ValueError, match=r"Setting custom parameter: Invalid argument\."):
        ddsim_device.submit_job("OPENQASM 3.0;", ProgramFormat.QASM3, 1, custom3="value")
    with pytest.raises(RuntimeError, match=r"Setting custom parameter: Not supported\."):
        ddsim_device.submit_job("OPENQASM 3.0;", ProgramFormat.QASM3, 1, custom4="value")
    with pytest.raises(RuntimeError, match=r"Setting custom parameter: Not supported\."):
        ddsim_device.submit_job("OPENQASM 3.0;", ProgramFormat.QASM3, 1, custom5="value")


@pytest.mark.parametrize("submission", ["text", "binary", "compiled", "source"])
def test_custom_parameter_bytes_preserve_seed(ddsim_device: Device, submission: str) -> None:
    """Packed native values reach direct and compiler submission unchanged."""
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; h q; c = measure q;'
    seed = struct.pack("@i", 1234567)
    if submission == "text":
        expected = ddsim_device.submit_job(source, ProgramFormat.QASM3, 64, custom1=1234567)
        actual = ddsim_device.submit_job(source, ProgramFormat.QASM3, 64, custom1=seed)
    elif submission == "binary":
        compiled = compile_program(source, output=OutputFormat.QIR_BASE)
        payload = compiled.to_bitcode()
        expected = ddsim_device.submit_job(payload, ProgramFormat.QIR_BASE_MODULE, 64, custom1=1234567)
        actual = ddsim_device.submit_job(payload, ProgramFormat.QIR_BASE_MODULE, 64, custom1=seed)
    elif submission == "compiled":
        compiled = compile_program(source, target=ddsim_device)
        expected = submit_program(compiled, target=ddsim_device, num_shots=64, custom1=1234567)
        actual = submit_program(compiled, target=ddsim_device, num_shots=64, custom1=seed)
    else:
        expected = submit_program(source, target=ddsim_device, num_shots=64, custom1=1234567)
        actual = submit_program(source, target=ddsim_device, num_shots=64, custom1=seed)
    expected.wait()
    actual.wait()
    assert actual.get_shots() == expected.get_shots()


def test_device_submit_job_without_shots(ddsim_device: Device) -> None:
    """Allow devices or custom programs to define their own repetitions."""
    qasm3_program = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""

    job = ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3)

    assert job.num_shots == 1024


def test_device_retrieve_job_by_id_reports_unsupported_provider(
    ddsim_device: Device,
) -> None:
    """Expose job retrieval through Python without requiring DDSIM support."""
    with pytest.raises(RuntimeError, match=r"Retrieving job: Not supported\."):
        ddsim_device.retrieve_job_by_id("unknown")


def test_empty_qasm_program_retains_zero_qubit_state(ddsim_device: Device) -> None:
    """Keep the zero-qubit amplitude distinct from an unavailable state."""
    program = "OPENQASM 3.0;"

    sample_job = ddsim_device.submit_job(program, ProgramFormat.QASM3, num_shots=4)
    sample_job.wait()
    assert sample_job.get_shots() == [""] * 4
    assert sample_job.get_counts() == {}
    assert sample_job.get_dense_statevector() == [1 + 0j]
    assert sample_job.get_sparse_statevector() == {"": 1 + 0j}

    state_job = ddsim_device.submit_job(program, ProgramFormat.QASM3, num_shots=0)
    state_job.wait()
    assert state_job.get_dense_statevector() == [1 + 0j]
    assert state_job.get_dense_probabilities() == [1.0]
    assert state_job.get_sparse_statevector() == {"": 1 + 0j}
    assert state_job.get_sparse_probabilities() == {"": 1.0}


def test_simulator_job_result_bindings(ddsim_device: Device) -> None:
    """Expose dense and sparse Bell-state results with Python container types."""
    job = ddsim_device.submit_job("OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];", ProgramFormat.QASM3, num_shots=0)
    assert job.wait()
    inv_sqrt2 = 1.0 / (2**0.5)

    state_vector = job.get_dense_statevector()
    assert state_vector.dtype == np.complex128
    assert state_vector.shape == (4,)
    assert state_vector.flags.c_contiguous
    assert state_vector == pytest.approx([inv_sqrt2, 0, 0, inv_sqrt2])

    probabilities = job.get_dense_probabilities()
    assert probabilities.dtype == np.float64
    assert probabilities.shape == (4,)
    assert probabilities == pytest.approx([0.5, 0, 0, 0.5])

    sparse_state_vector = job.get_sparse_statevector()
    assert isinstance(sparse_state_vector, dict)
    assert all(isinstance(value, complex) for value in sparse_state_vector.values())
    assert sparse_state_vector == pytest.approx({"00": inv_sqrt2, "11": inv_sqrt2})

    sparse_probabilities = job.get_sparse_probabilities()
    assert isinstance(sparse_probabilities, dict)
    assert all(isinstance(value, float) for value in sparse_probabilities.values())
    assert sparse_probabilities == pytest.approx({"00": 0.5, "11": 0.5})

    del job
    state_vector[0] = 0
    assert probabilities == pytest.approx([0.5, 0, 0, 0.5])
    assert state_vector == pytest.approx([0, 0, 0, inv_sqrt2])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Requires POSIX named pipes")
@pytest.mark.parametrize("entrypoint", ["driver", "builtin_driver", "compiler"])
def test_device_open_releases_gil(tmp_path: Path, entrypoint: str) -> None:
    """A Python thread can supply configuration while native opening waits."""
    script = """
import json
import os
import sys
from pathlib import Path
from threading import Thread

from mqt.core.mlir import CompilerTarget
from mqt.core.qdmi import builtin_driver
from mqt.core.qdmi import open_device

fifo = Path(sys.argv[1]) / "device.json"
os.mkfifo(fifo)
configuration = Path("json/sc/mqt-core-qdmi-sc-device.json").read_bytes()
os.environ["MQT_CORE_QDMI_CONFIG_JSON"] = json.dumps({
    "schema-version": 1,
    "qdmi": {"devices": [{
        "id": "mqt.sc.default",
        "session": {"device-config": {"file": str(fifo)}},
    }]},
})

def supply_configuration():
    # Opening the write end blocks until the native reader opens the FIFO.
    with fifo.open("wb") as stream:
        stream.write(configuration)

writer = Thread(target=supply_configuration, daemon=True)
writer.start()
entrypoint = sys.argv[2]
if entrypoint == "driver":
    assert open_device("mqt.sc.default").qubits_num() > 0
elif entrypoint == "builtin_driver":
    assert builtin_driver.open_device("mqt.sc.default").qubits_num() > 0
else:
    assert CompilerTarget.from_device_id("mqt.sc.default").num_sites > 0
writer.join()
"""
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script, str(tmp_path), entrypoint],
        check=True,
        timeout=15,
    )


def test_site_keeps_fresh_session_alive() -> None:
    """A site should remain usable after its device wrapper is destroyed."""
    site = open_device("mqt.sc.default").sites()[0]
    assert site.index() == 0


def test_operation_keeps_fresh_session_alive() -> None:
    """An operation should remain usable after its device wrapper is destroyed."""
    operation = open_device("mqt.sc.default").operations()[0]
    assert operation.name()
