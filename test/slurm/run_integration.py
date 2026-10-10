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
CLUSTER = ROOT / "examples" / "slurm"
DIST = ROOT / "dist"
RUNTIME = ROOT / "build" / "slurm-tests" / uuid.uuid4().hex
NODES: list[str] = []
COMPOSE = (
    "docker",
    "compose",
    "--project-name",
    f"mqt-core-slurm-{RUNTIME.name}",
    "--file",
    str(CLUSTER / "compose.yml"),
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
    return compose("exec", "-T", "--index", str(NODES.index(node) + 1), "node", *command, check=check)


def job(*command: str, check: bool = True, timeout: float = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Submit workloads as the same unprivileged user on every node."""
    return compose(
        "exec",
        "-T",
        "--user",
        "10000:10000",
        "controller",
        "env",
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
    run(("sh", str(CLUSTER / "prepare.sh"), str(RUNTIME)))


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
    for service, index, units in [
        ("controller", 1, ("munge.service", "slurmctld.service")),
        *(("node", index, ("munge.service", "slurmd.service")) for index in range(1, len(NODES) + 1)),
    ]:
        prefix = ("exec", "-T", "--index", str(index), service)
        compose(*prefix, "systemctl", "status", "--no-pager", *units, check=False, timeout=5)
        compose(
            *prefix,
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
        "from mqt.core.qdmi import device_ids; "
        "module_path = Path(mqt.core.__file__).resolve(); "
        "assert not any(module_path.is_relative_to(root) for root in ('/workspace', '/runtime')), module_path; "
        "ids = set(device_ids()); "
        "assert 'mqt.ddsim.default' in ids and 'mqt.sc.default' in ids, ids"
    )
    controller("python3", "-c", registry_check)
    assert_license("mqt.ddsim.default", total=2, used=0, free=2)
    assert_license("mqt.sc.default", total=1, used=0, free=1)

    compound = submit("ddsim-job.sh", "mqt.ddsim.default:1,mqt.sc.default:1")
    wait_for_failed_adapter(compound, "must name exactly one local QDMI device license")
    assert_license("mqt.ddsim.default", total=2, used=0, free=2)
    assert_license("mqt.sc.default", total=1, used=0, free=1)

    first = submit("ddsim-job.sh", "mqt.ddsim.default:1", node=NODES[0], hold=True)
    second = submit("ddsim-job.sh", "mqt.ddsim.default:1", node=NODES[1], hold=True)
    wait_for_result("ddsim", first, "the first DDSIM Bell result")
    wait_for_result("ddsim", second, "the second DDSIM Bell result")
    wait_for("the first DDSIM job to hold on its node", lambda: job_matches(first, "RUNNING", node=NODES[0]))
    wait_for("the second DDSIM job to hold on its node", lambda: job_matches(second, "RUNNING", node=NODES[1]))
    if "CPUAlloc=1" not in node_record(NODES[0]) or "CPUAlloc=1" not in node_record(NODES[1]):
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
    if sc_result["node"] not in NODES or sc_result["qubits"] <= 0:
        msg = f"Unexpected SC job result: {sc_result}"
        raise AssertionError(msg)
    if not job_matches(first, "RUNNING", node=NODES[0]) or not job_matches(second, "RUNNING", node=NODES[1]):
        msg = "The SC job did not complete while both DDSIM licenses remained held"
        raise AssertionError(msg)

    monitor_service = "qdmi-availability@mqt.ddsim.default:2.service"
    controller("systemctl", "stop", monitor_service)
    configuration = RUNTIME / "jobs" / "availability.json"
    configuration.write_text(
        json.dumps({"schema-version": 1, "qdmi": {"devices": [{"id": "mqt.ddsim.default", "enabled": False}]}}),
        encoding="utf-8",
    )
    monitor = ("python3", "/workspace/examples/slurm/update_availability.py", "--license", "mqt.ddsim.default:2")
    assert controller("env", "MQT_CORE_QDMI_CONFIG_FILE=/jobs/availability.json", *monitor, check=False).returncode == 1
    assert license_record("mqt.ddsim.default")["Reserved"] == "2"
    assert job_matches(first, "RUNNING")
    assert job_matches(second, "RUNNING")
    for job_id in (first, second):
        (RUNTIME / "jobs" / f"release-{job_id}").touch()
        wait_for(f"the released DDSIM job {job_id} to finish", lambda job_id=job_id: job_finished(job_id))
    assert_license("mqt.ddsim.default", total=2, used=0, free=2)
    controller(*monitor, "--block-only")
    job("srun", "--immediate=5", "--time=1", "--licenses=mqt.sc.default", "/bin/true", timeout=60)
    assert job_matches(third, "PENDING", node="", reason="Licenses")
    assert job("scontrol", "delete", "ReservationName=qdmi-unavailable-mqt.ddsim.default", check=False).returncode != 0

    controller("systemctl", "start", monitor_service)
    wait_for_result("ddsim", third, "the pending DDSIM job to execute after recovery")
    wait_for("the third DDSIM job to finish", lambda: job_finished(third))
    assert_bell_result(first, NODES[0])
    assert_bell_result(second, NODES[1])
    assert_bell_result(third)
    assert_license("mqt.ddsim.default", total=2, used=0, free=2)
    assert license_record("mqt.ddsim.default")["Reserved"] == "0"

    LOGGER.info(
        "Slurm enforced license capacity, kept unavailable-device jobs pending without allocating nodes, "
        "and resumed the pending Bell job after a successful health check."
    )


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    """Read the provider build inputs and the command to run in an allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=CLUSTER)
    parser.add_argument("--dist", type=Path, default=DIST)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--setup-script", default="")
    parser.add_argument("--compose-file", type=Path)
    parser.add_argument("--device-license")
    parser.add_argument("--qdmi-config-file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    if options.nodes < 2:
        parser.error("--nodes must be at least two for the admission tests")
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
    return options


def test_provider(options: argparse.Namespace) -> None:
    """Execute a device workload with its submission environment."""
    environment = [f"MQT_CORE_QDMI_CONFIG_FILE={options.qdmi_config_file}"] if options.qdmi_config_file else []
    allocation = ("srun", "--immediate=5", "--time=5", "--ntasks=1", f"--licenses={options.device_license}:1")
    job("env", *environment, *allocation, *options.command, timeout=300)


def test_job_environment() -> None:
    """Export job settings through srun and sbatch independently of daemon settings."""
    program = (
        "import os; assert os.geteuid() == 10000; "
        "assert os.environ['MQT_CORE_QDMI_CONFIG_FILE'] == '/jobs/devices.json'; "
        "assert os.environ['MQT_SLURM_TEST_REFERENCE'] == 'job-value'"
    )
    environment = ("env", "MQT_CORE_QDMI_CONFIG_FILE=/jobs/devices.json", "MQT_SLURM_TEST_REFERENCE=job-value")
    job(*environment, "srun", "--immediate=5", "--time=1", "python3", "-c", program, timeout=60)
    job(
        *environment,
        "sbatch",
        "--wait",
        "--time=1",
        "--output=/jobs/environment.out",
        "--wrap",
        shlex.join(("python3", "-c", program)),
        timeout=120,
    )
    job(
        "srun",
        "--immediate=5",
        "--time=1",
        "python3",
        "-c",
        "import os; assert 'MQT_SLURM_TEST_REFERENCE' not in os.environ; "
        "assert 'MQT_CORE_QDMI_CONFIG_FILE' not in os.environ",
        timeout=60,
    )


def test_explicit_check() -> None:
    """Run the payload only after the checker accepts the job's configuration."""
    body = RUNTIME / "jobs" / "checked-body"
    (RUNTIME / "jobs" / "checker.qdmi.json").write_text(
        json.dumps({"schema-version": 1, "qdmi": {"devices": [{"id": "mqt.sc.default", "enabled": False}]}}),
        encoding="utf-8",
    )
    for enabled in (True, False):
        body.unlink(missing_ok=True)
        setup = "unset MQT_CORE_QDMI_CONFIG_FILE MQT_CORE_QDMI_CONFIG_JSON\n"
        if not enabled:
            setup += "export MQT_CORE_QDMI_CONFIG_FILE=/jobs/checker.qdmi.json\n"
        result = job(
            "srun",
            "--immediate=5",
            "--time=1",
            "--ntasks=1",
            "--licenses=mqt.sc.default",
            "sh",
            "-ec",
            setup + "mqt-core-qdmi-check --device mqt.sc.default\ntouch /jobs/checked-body",
            check=False,
            timeout=60,
        )
        assert (result.returncode == 0) == enabled
        assert body.exists() == enabled


def main(arguments: Sequence[str] = (), *, workload: Callable[[], None] | None = None) -> None:
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
        if options.qdmi_config_file:
            (RUNTIME / "qdmi.env").write_text(f"MQT_CORE_QDMI_CONFIG_FILE={options.qdmi_config_file}\n")
        LOGGER.info("Slurm runtime directory: %s", RUNTIME)
        started = True
        compose(
            "up",
            "--build",
            "--detach",
            "--scale",
            f"node={options.nodes}",
            "--wait",
            "--wait-timeout",
            "120",
            timeout=1800,
            capture_output=False,
        )
        NODES[:] = [
            compose("exec", "-T", "--index", str(index), "node", "hostname").stdout.strip()
            for index in range(1, options.nodes + 1)
        ]
        LOGGER.info("Slurm image build and startup: %.2fs", time.monotonic() - started_at)
        testing_at = time.monotonic()

        version_output = controller("scontrol", "--version").stdout.strip()
        version_match = re.search(r"^slurm(?:-wlm)?\s+(\d+)\.(\d+)\b", version_output, flags=re.IGNORECASE)
        if version_match is None or tuple(map(int, version_match.groups())) < (25, 11):
            msg = f"The fixture requires Slurm 25.11 or newer, got {version_output!r}"
            raise RuntimeError(msg)

        for node in NODES:
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
            wait_for(f"{node} to become IDLE with two processors", lambda node=node: node_is_idle(node))

        registered = set(controller("sinfo", "--Node", "--noheader", "--format=%N").stdout.split())
        assert registered == set(NODES), registered
        healthy_licenses = (
            "mqt.ddsim.default",
            "mqt.sc.default",
            *([options.device_license] if options.device_license else []),
        )
        for license_name in healthy_licenses:
            wait_for(
                f"{license_name} to become available",
                lambda license_name=license_name: license_record(license_name)["Reserved"] == "0",
            )
        if workload is not None:
            workload()
        elif options.command:
            test_provider(options)
        else:
            test_core()
            test_job_environment()
            test_explicit_check()

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
