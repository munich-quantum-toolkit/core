# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run the real Slurm admission and QDMI execution integration test."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test" / "slurm"
DIST = FIXTURE / "dist"
RUNTIME = FIXTURE / "runtime" / uuid.uuid4().hex
COMPOSE = (
    "docker",
    "compose",
    "--project-name",
    f"mqt-core-slurm-{RUNTIME.name}",
    "--file",
    str(FIXTURE / "compose.yml"),
)
TIMEOUT = 120.0
COMMAND_TIMEOUT = 30.0
RESULT_VISIBILITY_GRACE_PERIOD = 5.0
LOGGER = logging.getLogger(__name__)
COMPOSE_FILES: list[str] = []
COMPOSE_ENV: dict[str, str] = {}


def _communicate(process: subprocess.Popen[str], timeout: float) -> tuple[str, str]:
    """Collect output, killing the whole command group on interruption.

    Returns:
        The command's stdout and stderr.
    """
    try:
        return process.communicate(timeout=timeout)
    except BaseException:
        # Docker can spawn a Compose child holding our output pipes.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise


def run(
    command: Sequence[str],
    *,
    check: bool = True,
    timeout: float = COMMAND_TIMEOUT,
    capture_output: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and retain output for assertions and diagnostics."""
    LOGGER.info("+ %s", shlex.join(command))
    try:
        with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            env=env,
            start_new_session=True,
        ) as process:
            stdout, stderr = _communicate(process, timeout)
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        if check:
            raise
        LOGGER.warning("Command failed: %s", error)
        return subprocess.CompletedProcess(
            command, 124 if isinstance(error, subprocess.TimeoutExpired) else 127, "", str(error)
        )
    if result.stdout:
        LOGGER.info("%s", result.stdout.rstrip())
    if result.stderr:
        LOGGER.info("%s", result.stderr.rstrip())
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
    return result


def compose(
    *arguments: str,
    check: bool = True,
    timeout: float = COMMAND_TIMEOUT,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run Docker Compose for this invocation's isolated project and artifacts."""
    return run(
        (*COMPOSE, *COMPOSE_FILES, *arguments),
        check=check,
        timeout=timeout,
        capture_output=capture_output,
        env={**os.environ, **COMPOSE_ENV, "MQT_CORE_SLURM_RUNTIME": str(RUNTIME)},
    )


def controller(*command: str, check: bool = True, timeout: float = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Run a Slurm client command in the controller container."""
    return compose("exec", "-T", "controller", *command, check=check, timeout=timeout)


def compute(node: str, *command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a diagnostic command in one compute container."""
    return compose("exec", "-T", node, *command, check=check)


def job(*command: str, check: bool = True, timeout: float = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Submit workloads as the same unprivileged user on every node."""
    return compose(
        "exec",
        "-T",
        "--user",
        "10000:10000",
        "controller",
        "env",
        "PYTHONPATH=/workspace/test/slurm",
        *command,
        check=check,
        timeout=timeout,
    )


def wait_for(description: str, predicate: Callable[[], bool], timeout: float = TIMEOUT) -> None:
    """Poll a condition and report a precise timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    msg = f"Timed out while waiting for {description}"
    raise TimeoutError(msg)


def node_record(node: str) -> str:
    """Return one machine-readable Slurm node record."""
    return controller("scontrol", "show", "node", node, "--oneliner").stdout.strip()


def node_is_idle(node: str) -> bool:
    """Return whether a compute node has registered and is idle."""
    record = node_record(node)
    return "State=IDLE" in record and "CPUTot=2" in record


def job_record(job_id: str) -> tuple[str, str, str] | None:
    """Return state, reason, and node for an active job."""
    output = controller("squeue", "--noheader", "--jobs", job_id, "--format=%T|%R|%N").stdout.strip()
    if not output:
        return None
    state, reason, node = output.split("|", maxsplit=2)
    # squeue renders a pending reason in parentheses even with a custom format.
    if reason.startswith("(") and reason.endswith(")"):
        reason = reason[1:-1]
    return state, reason, node


def job_matches(job_id: str, state: str, *, node: str | None = None, reason: str | None = None) -> bool:
    """Return whether an active job has the requested observable state."""
    record = job_record(job_id)
    if record is None:
        return False
    actual_state, actual_reason, actual_node = record
    return (
        actual_state == state and (node is None or actual_node == node) and (reason is None or actual_reason == reason)
    )


def job_finished(job_id: str, *, expected_state: str = "COMPLETED") -> bool:
    """Require the terminal state and exit code, without an accounting daemon."""
    output = controller("scontrol", "show", "job", job_id, "--oneliner").stdout
    record = dict(field.split("=", maxsplit=1) for field in output.split() if "=" in field)
    state = record["JobState"]
    if state in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}:
        return False
    exit_code = tuple(map(int, record["ExitCode"].split(":")))
    if state != expected_state or len(exit_code) != 2 or ((exit_code == (0, 0)) != (expected_state == "COMPLETED")):
        msg = f"Slurm job {job_id} ended with {state}, ExitCode={record['ExitCode']}; expected {expected_state}"
        raise AssertionError(msg)
    return True


def submit(script: str, license_expression: str, *, node: str | None = None, hold: bool = False) -> str:
    """Submit one single-processor batch job and return its numeric ID."""
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--time=5",
        f"--licenses={license_expression}",
        "--chdir=/workspace",
        "--output=/jobs/slurm-%j.out",
    ]
    if node is not None:
        command.append(f"--nodelist={node}")
    command.append(f"/workspace/test/slurm/{script}")
    if hold:
        command.append("--hold")
    job_id = job(*command).stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdecimal():
        msg = f"sbatch returned an invalid job ID: {job_id!r}"
        raise RuntimeError(msg)
    return job_id


def license_record(name: str) -> dict[str, str]:
    """Parse one `scontrol show lic` record."""
    output = controller("scontrol", "show", "lic", name, "--oneliner").stdout
    return dict(field.split("=", maxsplit=1) for field in output.split() if "=" in field)


def assert_license(name: str, *, total: int, used: int, free: int) -> None:
    """Require the exact static Slurm license counters."""
    record = license_record(name)
    expected = {"LicenseName": name, "Total": str(total), "Used": str(used), "Free": str(free)}
    if not expected.items() <= record.items():
        msg = f"Unexpected license record for {name}: {record}; expected {expected}"
        raise AssertionError(msg)


def load_result(kind: str, job_id: str) -> dict[str, Any]:
    """Load one batch-job result from the shared runtime directory."""
    return json.loads((RUNTIME / "jobs" / f"{kind}-{job_id}.json").read_text(encoding="utf-8"))


def wait_for_result(kind: str, job_id: str, description: str) -> None:
    """Wait for a result and allow bounded shared-file visibility delay."""
    result_path = RUNTIME / "jobs" / f"{kind}-{job_id}.json"
    left_queue_at: float | None = None

    def result_exists_or_raise() -> bool:
        nonlocal left_queue_at
        if result_path.exists():
            return True
        if job_record(job_id) is not None:
            left_queue_at = None
            return False

        now = time.monotonic()
        if left_queue_at is None:
            left_queue_at = now
            return False
        if now - left_queue_at < RESULT_VISIBILITY_GRACE_PERIOD:
            return False

        output_path = RUNTIME / "jobs" / f"slurm-{job_id}.out"
        output = output_path.read_text(encoding="utf-8") if output_path.exists() else "<no batch output>"
        msg = f"Slurm job {job_id} exited before producing {result_path.name}:\n{output.rstrip()}"
        raise RuntimeError(msg)

    wait_for(description, result_exists_or_raise)


def wait_for_failed_adapter(job_id: str, diagnostic: str) -> None:
    """Require an adapter diagnostic and a failed batch job without a result."""
    output_path = RUNTIME / "jobs" / f"slurm-{job_id}.out"

    def failed_with_diagnostic() -> bool:
        if not job_finished(job_id, expected_state="FAILED") or not output_path.exists():
            return False
        return diagnostic in output_path.read_text(encoding="utf-8")

    wait_for(f"Slurm job {job_id} to fail with {diagnostic!r}", failed_with_diagnostic)
    if (RUNTIME / "jobs" / f"ddsim-{job_id}.json").exists():
        msg = f"Rejected Slurm job {job_id} unexpectedly produced a DDSIM result"
        raise AssertionError(msg)


def assert_bell_result(job_id: str, expected_node: str | None = None) -> None:
    """Recheck the Bell result outside the batch job."""
    result = load_result("ddsim", job_id)
    if result["shots"] != 256 or sum(result["counts"].values()) != 256:
        msg = f"Slurm job {job_id} did not return 256 Bell samples: {result}"
        raise AssertionError(msg)
    if set(result["counts"]) != {"00", "11"}:
        msg = f"Slurm job {job_id} returned non-Bell outcomes: {result}"
        raise AssertionError(msg)
    if expected_node is not None and result["node"] != expected_node:
        msg = f"Slurm job {job_id} ran on {result['node']}, expected {expected_node}"
        raise AssertionError(msg)
    if result["licenses"] not in {"mqt.ddsim.default", "mqt.ddsim.default:1"}:
        msg = f"Slurm exposed an unexpected license string: {result['licenses']}"
        raise AssertionError(msg)


def clean_runtime() -> None:
    """Create private artifacts and a Munge key for this invocation only."""
    RUNTIME.mkdir(mode=0o700, parents=True, exist_ok=False)
    key = RUNTIME / "munge.key"
    key.write_bytes(secrets.token_bytes(1024))
    key.chmod(0o600)
    (RUNTIME / "jobs").mkdir()
    (RUNTIME / "jobs").chmod(0o1777)
    shutil.copyfile(FIXTURE / "slurm.conf", RUNTIME / "slurm.conf")
    (RUNTIME / "plugstack.conf").touch()


def print_diagnostics() -> None:
    """Print cluster state without hiding the original test failure."""
    controller("squeue", "--all", check=False, timeout=5)
    controller(
        "sacct",
        "--allusers",
        "--starttime=now-1hour",
        "--format=JobID,State,ExitCode,Reason,NodeList",
        check=False,
        timeout=5,
    )
    controller("scontrol", "show", "node", check=False, timeout=5)
    controller("scontrol", "show", "lic", check=False, timeout=5)
    for output in sorted((RUNTIME / "jobs").glob("slurm-*.out")):
        LOGGER.info("=== %s ===", output.name)
        try:
            LOGGER.info("%s", output.read_text(encoding="utf-8").rstrip())
        except OSError as error:
            LOGGER.info("Could not read %s: %s", output, error)
    for service, units in (
        ("controller", ("munge.service", "slurmctld.service")),
        ("node1", ("munge.service", "slurmd.service")),
        ("node2", ("munge.service", "slurmd.service")),
    ):
        compose("exec", "-T", service, "systemctl", "status", "--no-pager", *units, check=False, timeout=5)
        compose(
            "exec",
            "-T",
            service,
            "journalctl",
            "--no-pager",
            "--lines=100",
            *(argument for unit in units for argument in ("--unit", unit)),
            check=False,
            timeout=5,
        )
    compose("logs", "--no-color", check=False, timeout=5)


def test_core() -> None:
    """Verify admission and execution with the bundled DDSIM and SC devices."""
    registry_check = (
        "from pathlib import Path; "
        "import mqt.core; "
        "from mqt.core.qdmi import driver; "
        "module_path = Path(mqt.core.__file__).resolve(); "
        "assert not any(module_path.is_relative_to(root) for root in ('/workspace', '/runtime')), module_path; "
        "ids = driver.registered_device_ids(); "
        "assert 'mqt.ddsim.default' in ids and 'mqt.sc.default' in ids, ids"
    )
    controller("python3", "-c", registry_check)
    assert_license("mqt.ddsim.default", total=2, used=0, free=2)
    assert_license("mqt.sc.default", total=1, used=0, free=1)

    non_unit = submit("ddsim-job.sh", "mqt.ddsim.default:2")
    wait_for_failed_adapter(non_unit, "must request exactly one Slurm license")
    compound = submit("ddsim-job.sh", "mqt.ddsim.default:1,mqt.sc.default:1")
    wait_for_failed_adapter(compound, "uses a compound AND expression")
    alternative = submit("ddsim-job.sh", "mqt.ddsim.default:1|mqt.sc.default:1")
    wait_for_failed_adapter(alternative, "uses a compound OR expression")
    assert_license("mqt.ddsim.default", total=2, used=0, free=2)
    assert_license("mqt.sc.default", total=1, used=0, free=1)

    first = submit("ddsim-job.sh", "mqt.ddsim.default:1", node="node1", hold=True)
    second = submit("ddsim-job.sh", "mqt.ddsim.default:1", node="node2", hold=True)
    wait_for_result("ddsim", first, "the first DDSIM Bell result")
    wait_for_result("ddsim", second, "the second DDSIM Bell result")
    wait_for("the first DDSIM job to hold on node1", lambda: job_matches(first, "RUNNING", node="node1"))
    wait_for("the second DDSIM job to hold on node2", lambda: job_matches(second, "RUNNING", node="node2"))
    if "CPUAlloc=1" not in node_record("node1") or "CPUAlloc=1" not in node_record("node2"):
        msg = "Each held DDSIM job must leave one processor free on its compute node"
        raise AssertionError(msg)

    third = submit("ddsim-job.sh", "mqt.ddsim.default:1")
    wait_for(
        "the third DDSIM job to wait for its license",
        lambda: job_matches(third, "PENDING", reason="Licenses"),
    )
    assert_license("mqt.ddsim.default", total=2, used=2, free=0)

    sc_job = submit("sc-job.sh", "mqt.sc.default:1")
    wait_for_result("sc", sc_job, "the SC job to execute on a free CPU")
    wait_for("the SC job to complete", lambda: job_finished(sc_job))
    sc_result = load_result("sc", sc_job)
    if sc_result["node"] not in {"node1", "node2"} or sc_result["qubits"] <= 0:
        msg = f"Unexpected SC job result: {sc_result}"
        raise AssertionError(msg)
    if not job_matches(first, "RUNNING", node="node1") or not job_matches(second, "RUNNING", node="node2"):
        msg = "The SC job did not complete while both DDSIM licenses remained held"
        raise AssertionError(msg)

    (RUNTIME / "jobs" / f"release-{first}").touch()
    wait_for("the released first DDSIM job to finish", lambda: job_finished(first))
    wait_for_result("ddsim", third, "the pending third DDSIM job to execute")
    wait_for("the third DDSIM job to finish", lambda: job_finished(third))

    assert_bell_result(first, "node1")
    assert_bell_result(second, "node2")
    assert_bell_result(third)
    assert_license("mqt.ddsim.default", total=2, used=1, free=1)

    (RUNTIME / "jobs" / f"release-{second}").touch()
    wait_for("the released second DDSIM job to finish", lambda: job_finished(second))
    assert_license("mqt.ddsim.default", total=2, used=0, free=2)

    LOGGER.info(
        "Slurm 25.11+ admitted two held DDSIM jobs, blocked the third for Licenses, "
        "ran the SC job on a free CPU, and executed the third Bell job after release."
    )


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    """Read the provider build inputs and the command to run in an allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT)
    parser.add_argument("--dist", type=Path, default=DIST)
    parser.add_argument("--setup-script", default="")
    parser.add_argument("--compose-file", type=Path)
    parser.add_argument("--device-license")
    parser.add_argument("--qdmi-config-file")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    if options.command[:1] == ["--"]:
        options.command = options.command[1:]
    if bool(options.command) != bool(options.device_license):
        parser.error("--device-license and a command after -- must be supplied together")
    if options.device_license and re.search(r"[\s,:|@]", options.device_license):
        parser.error("--device-license must be one local device ID without a count")
    if options.setup_script:
        setup = (options.workload / options.setup_script).resolve()
        if not setup.is_relative_to(options.workload.resolve()) or not setup.is_file():
            parser.error("--setup-script must name a file inside --workload")
    for reference in options.reference:
        name, separator, value = reference.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None or any(c.isspace() for c in value):
            parser.error("--reference must use ENV=value without whitespace")
    if options.qdmi_config_file and any(c.isspace() for c in options.qdmi_config_file):
        parser.error("--qdmi-config-file must not contain whitespace")
    return options


def test_provider(options: argparse.Namespace) -> None:
    """Run the same provider workload with direct configuration and SPANK."""
    environment = list(options.reference)
    if options.qdmi_config_file:
        environment.append(f"MQT_CORE_QDMI_CONFIG_FILE={options.qdmi_config_file}")
    allocation = ("srun", "--immediate=5", "--time=5", "--ntasks=1", f"--licenses={options.device_license}:1")
    job("env", *environment, *allocation, *options.command, timeout=300)

    configuration = [
        "required /usr/local/lib/slurm/mqt-core-qdmi-spank.so",
        f"licenses={options.device_license}",
    ]
    if options.qdmi_config_file:
        configuration.append(f"qdmi_config_file={options.qdmi_config_file}")
    for reference in options.reference:
        name, _, value = reference.partition("=")
        configuration.append(f"reference={name}:{options.device_license}:{value}")
    (RUNTIME / "plugstack.conf").write_text(" ".join(configuration) + "\n", encoding="utf-8")
    job(*allocation, *options.command, timeout=300)
    (RUNTIME / "plugstack.conf").write_text("", encoding="utf-8")


def test_spank_transport() -> None:
    """Check shared option transport through both srun and sbatch."""
    selected = "mqt.ddsim.default"
    other = "mqt.sc.default"
    reference = "MQT_SLURM_TEST_REFERENCE"
    second_reference = "MQT_SLURM_TEST_SECOND_REFERENCE"
    catalogue = "/runtime/site.qdmi.json"
    (RUNTIME / "plugstack.conf").write_text(
        "required /usr/local/lib/slurm/mqt-core-qdmi-spank.so "
        f"licenses={selected},{other} qdmi_config_file={catalogue} "
        f"reference={reference}:{selected}:site-default "
        f"reference={second_reference}:{selected}:\n",
        encoding="utf-8",
    )
    program = (
        "import json, os; assert os.geteuid() == 10000, os.geteuid(); "
        f"print(json.dumps([os.environ.get('{reference}'), os.environ.get('MQT_CORE_QDMI_CONFIG_FILE')]), flush=True)"
    )
    allocation = ("srun", "--immediate=5", "--time=1", "--ntasks=1")

    def values(*arguments: str) -> list[str | None]:
        return json.loads(job(*arguments, "python3", "-c", program, timeout=60).stdout)

    assert values(*allocation, f"--licenses={selected}") == ["site-default", catalogue]
    assert values("env", f"{reference}=job-value", *allocation, f"--licenses={selected}") == ["job-value", catalogue]
    assert values("env", "MQT_CORE_QDMI_CONFIG_FILE=/runtime/job.qdmi.json", *allocation, f"--licenses={selected}") == [
        "site-default",
        "/runtime/job.qdmi.json",
    ]
    assert values(
        "env",
        f"{reference}=job-value",
        "MQT_CORE_QDMI_CONFIG_FILE=/runtime/job.qdmi.json",
        *allocation,
        f"--licenses={selected}",
        f"--qdmi-ref-{reference}=option-value",
        "--qdmi-config-file=/runtime/option.qdmi.json",
    ) == ["option-value", "/runtime/option.qdmi.json"]
    assert values(*allocation, f"--licenses={other}") == [None, catalogue]
    assert values(*allocation) == [None, None]

    # Slurm allocation metadata is not subject to the QDMI reference size limit.
    unrelated = ",".join(f"unrelated-{index:03d}-{'x' * 49}:1" for index in range(65))
    slurm_config = RUNTIME / "slurm.conf"
    original_config = slurm_config.read_text(encoding="utf-8")
    slurm_config.write_text(original_config.replace("Licenses=", f"Licenses={unrelated},", 1), encoding="utf-8")
    controller("scontrol", "reconfigure")
    try:
        assert values(*allocation, f"--licenses={unrelated}") == [None, None]
        assert values(*allocation, f"--licenses={unrelated},{selected}") == ["site-default", catalogue]
    finally:
        slurm_config.write_text(original_config, encoding="utf-8")
        controller("scontrol", "reconfigure")

    for options in (
        (f"--qdmi-ref-{reference}=not-allocated",),
        (f"--licenses={selected}", "--qdmi-ref-UNLISTED=value"),
        (f"--licenses={other}", f"--qdmi-ref-{reference}=wrong-device"),
    ):
        result = job(*allocation, *options, "/bin/true", check=False, timeout=60)
        assert result.returncode != 0, options

    for value in ("", "multiline\nvalue", "x" * 4096):
        for command in (
            (*allocation, f"--licenses={selected}", f"--qdmi-ref-{reference}={value}", "/bin/true"),
            ("env", f"{reference}={value}", *allocation, f"--licenses={selected}", "/bin/true"),
        ):
            result = job(*command, check=False, timeout=60)
            assert result.returncode != 0, "Malformed reference unexpectedly reached the task"
    for node in ("node1", "node2"):
        wait_for(f"{node} to return to IDLE after rejected tasks", lambda node=node: node_is_idle(node))
        assert "DRAIN" not in node_record(node)

    # A submitted environment value must not turn an unlicensed allocation into
    # a matching job in the remote SPANK hook.
    assert values("env", f"SLURM_JOB_LICENSES={selected}", *allocation) == [None, None]

    result = job(
        *allocation,
        f"--licenses={selected}",
        f"--qdmi-ref-{reference}=first",
        f"--qdmi-ref-{second_reference}=second",
        "python3",
        "-c",
        f"import os; assert os.environ['{reference}'] == 'first'; assert os.environ['{second_reference}'] == 'second'",
        timeout=60,
    )
    assert result.returncode == 0

    output = RUNTIME / "jobs" / "spank-batch.out"
    release = RUNTIME / "jobs" / "spank-batch-release"
    batch = (
        job(
            "sbatch",
            "--parsable",
            "--time=1",
            "--ntasks=1",
            "--nodelist=node1",
            f"--licenses={selected}",
            f"--qdmi-ref-{reference}=batch-value",
            "--output=/jobs/spank-batch.out",
            "--wrap",
            shlex.join((
                "python3",
                "-c",
                program + "; import time; from pathlib import Path\n"
                "while not Path('/jobs/spank-batch-release').exists(): time.sleep(0.1)",
            )),
        )
        .stdout.strip()
        .split(";", maxsplit=1)[0]
    )
    try:
        wait_for("the SPANK batch task to start", lambda: output.exists() and bool(output.read_text(encoding="utf-8")))
        assert json.loads(output.read_text(encoding="utf-8")) == ["batch-value", catalogue]
        compute(
            "node1",
            "python3",
            "-c",
            "from pathlib import Path\n"
            "seen = set()\n"
            "for process in Path('/proc').glob('[0-9]*'):\n"
            "    try:\n"
            "        name = (process / 'comm').read_text().strip()\n"
            "        if name not in {'slurmd', 'slurmstepd'}: continue\n"
            "        entries = (process / 'environ').read_bytes().split(b'\\0')\n"
            "        environment = dict(entry.split(b'=', 1) for entry in entries if b'=' in entry)\n"
            "    except FileNotFoundError:\n"
            "        continue\n"
            "    seen.add(name)\n"
            f"    assert environment.get(b'{reference}') in (None, b'daemon-only'), name\n"
            "    assert environment.get(b'MQT_CORE_QDMI_CONFIG_FILE') in (None, b'/daemon-only/qdmi.json'), name\n"
            "assert seen == {'slurmd', 'slurmstepd'}, seen\n",
        )
    finally:
        release.touch()
    wait_for("the SPANK batch job to complete", lambda: job_finished(batch))
    (RUNTIME / "plugstack.conf").write_text("", encoding="utf-8")


def main(arguments: Sequence[str] = ()) -> None:
    """Build the shared cluster and run its Core or provider workload."""
    options = parse_arguments(arguments)
    wheels = tuple(options.dist.glob("mqt_core-*.whl"))
    if len(wheels) != 1:
        msg = f"Build exactly one MQT Core wheel in {options.dist}, found {len(wheels)}"
        raise RuntimeError(msg)
    COMPOSE_ENV.update({
        "MQT_CORE_SLURM_CORE": str(ROOT),
        "MQT_CORE_SLURM_WORKLOAD": str(options.workload.resolve()),
        "MQT_CORE_SLURM_DIST": str(options.dist.resolve()),
        "MQT_CORE_SLURM_SETUP_SCRIPT": options.setup_script,
    })
    COMPOSE_FILES[:] = ("--file", str(options.compose_file.resolve())) if options.compose_file else ()

    started_at = time.monotonic()

    success = False
    started = False
    try:
        cgroup_version = run(("docker", "info", "--format", "{{.CgroupVersion}}")).stdout.strip()
        if cgroup_version != "2":
            msg = f"The Slurm integration requires Docker on cgroup v2, got {cgroup_version!r}"
            raise RuntimeError(msg)

        clean_runtime()
        if options.device_license:
            configuration = (RUNTIME / "slurm.conf").read_text(encoding="utf-8")
            configuration = re.sub(
                r"^Licenses=(.*)$", rf"Licenses=\1,{options.device_license}:2", configuration, flags=re.MULTILINE
            )
            (RUNTIME / "slurm.conf").write_text(configuration, encoding="utf-8")
        LOGGER.info("Slurm runtime directory: %s", RUNTIME)
        started = True
        compose("up", "--build", "--detach", "--wait", "--wait-timeout", "120", timeout=1800, capture_output=False)
        LOGGER.info("Slurm image build and startup: %.2fs", time.monotonic() - started_at)
        testing_at = time.monotonic()

        version_output = controller("scontrol", "--version").stdout.strip()
        version_match = re.search(r"^slurm(?:-wlm)?\s+(\d+)\.(\d+)\b", version_output, flags=re.IGNORECASE)
        if version_match is None or tuple(map(int, version_match.groups())) < (25, 11):
            msg = f"The fixture requires Slurm 25.11 or newer, got {version_output!r}"
            raise RuntimeError(msg)

        for node in ("node1", "node2"):
            compute(node, "test", "-r", "/sys/fs/cgroup/cgroup.controllers")
            delegate = compute(
                node,
                "systemctl",
                "show",
                "slurmd.service",
                "--property=Delegate",
                "--value",
            ).stdout.strip()
            if delegate != "yes":
                msg = f"The packaged slurmd.service on {node} must set Delegate=yes, got {delegate!r}"
                raise RuntimeError(msg)
            compute(
                node,
                "python3",
                "-c",
                "from pathlib import Path; import subprocess; "
                "pid = subprocess.check_output(['systemctl', 'show', 'slurmd.service', "
                "'--property=MainPID', '--value'], text=True).strip(); "
                "environment = Path('/proc/' + pid + '/environ').read_bytes().split(b'\\0'); "
                "assert b'MQT_SLURM_TEST_REFERENCE=daemon-only' in environment; "
                "assert b'MQT_CORE_QDMI_CONFIG_FILE=/daemon-only/qdmi.json' in environment",
            )
            wait_for(f"{node} to become IDLE with two processors", lambda node=node: node_is_idle(node))

        if options.command:
            test_provider(options)
        else:
            test_core()
        test_spank_transport()

        success = True
        LOGGER.info("Slurm admission and execution checks: %.2fs", time.monotonic() - testing_at)
    finally:
        try:
            if started and not success:
                try:
                    print_diagnostics()
                except Exception:
                    LOGGER.exception("Could not collect Slurm diagnostics")
        finally:
            if started:
                stopped = compose("down", "--volumes", "--remove-orphans", "--rmi", "local", check=False, timeout=60)
                if success and stopped.returncode == 0:
                    shutil.rmtree(RUNTIME)
                elif success:
                    msg = f"Slurm cleanup failed; retained artifacts in {RUNTIME}"
                    raise RuntimeError(msg)
            LOGGER.info("Slurm integration total: %.2fs", time.monotonic() - started_at)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
