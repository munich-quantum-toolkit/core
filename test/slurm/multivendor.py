# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exercise IQM and Braket independently on one credentialed Slurm cluster."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
from pathlib import Path

import run_integration as cluster

DEVICES = {"iqm": "iqm.emerald.mock", "braket": "amazon.braket.sv1"}
AWS_CREDENTIALS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def environment(vendor: str) -> tuple[str, ...]:
    """Select one runtime and remove the other vendor's credentials before submission."""
    unrelated = AWS_CREDENTIALS if vendor == "iqm" else ("IQM_TOKEN", "IQM_TOKENS_FILE")
    return (
        "env",
        *(argument for name in unrelated for argument in ("-u", name)),
        f"PATH=/opt/runtimes/{vendor}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"MQT_CORE_QDMI_CONFIG_FILE=/jobs/{vendor}.json",
    )


def submit(vendor: str, *, probe: bool = True) -> str:
    """Submit one licensed batch job, retaining the allocation after a successful probe."""
    unrelated = AWS_CREDENTIALS if vendor == "iqm" else ("IQM_TOKEN", "IQM_TOKENS_FILE")
    metadata = (
        "import json, os, socket; from pathlib import Path; "
        f"assert os.environ['SLURM_JOB_LICENSES'] in ('{DEVICES[vendor]}', '{DEVICES[vendor]}:1'); "
        f"assert not any(os.environ.get(key) for key in {unrelated!r}); "
        f"Path('/jobs/{vendor}-' + os.environ['SLURM_JOB_ID'] + '.json').write_text("
        "json.dumps({'node': socket.gethostname(), 'licenses': os.environ['SLURM_JOB_LICENSES']}))"
    )
    body = "set -eu\n"
    if probe:
        body += f"mqt-core-qdmi-check --device {DEVICES[vendor]} --timeout 30\npython3 /{vendor}/test/slurm/probe.py\n"
    body += f"python3 -c {shlex.quote(metadata)}\n"
    if probe:
        body += 'while [ ! -f "/jobs/release-$SLURM_JOB_ID" ]; do sleep 0.2; done\n'
    result = cluster.job(
        *environment(vendor),
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--time=5",
        f"--licenses={DEVICES[vendor]}",
        "--output=/jobs/slurm-%j.out",
        "--wrap",
        body,
    )
    submitted = result.stdout.strip().split(";", maxsplit=1)[0]
    if not submitted.isdecimal():
        msg = f"sbatch returned an invalid job ID: {submitted!r}"
        raise RuntimeError(msg)
    return submitted


def test_vendors() -> None:
    """Keep vendor allocations independent while one device license is unavailable."""
    configuration = cluster.RUNTIME / "slurm.conf"
    licenses = ",".join(f"{device}:1" for device in DEVICES.values())
    configuration.write_text(configuration.read_text().replace("Licenses=", f"Licenses={licenses},", 1))
    cluster.controller("scontrol", "reconfigure")
    for vendor, device in DEVICES.items():
        cluster.controller(
            *environment(vendor),
            "PROVIDER_INSTALL_MODE=wheel",
            "sh",
            f"/{vendor}/test/slurm/setup.sh",
            f"/jobs/{vendor}.json",
        )
        other = "amazon.braket." if vendor == "iqm" else "iqm."
        cluster.job(
            *environment(vendor),
            "python3",
            "-c",
            "from mqt.core.qdmi.builtin_driver import registered_device_ids; "
            f"ids = registered_device_ids(); assert '{device}' in ids; "
            f"assert not any(device.startswith('{other}') for device in ids)",
        )

    iqm, braket = (submit(vendor) for vendor in DEVICES)
    for vendor, job_id in (("iqm", iqm), ("braket", braket)):
        cluster.wait_for_result(vendor, job_id, f"the {vendor} eight-shot workload")
    for vendor, job_id in (("iqm", iqm), ("braket", braket)):
        assert cluster.job_matches(job_id, "RUNNING")
        cluster.assert_license(DEVICES[vendor], total=1, used=1, free=0)
        assert cluster.load_result(vendor, job_id)["node"] in cluster.NODES

    monitor = (
        "python3",
        "/workspace/examples/slurm/update_availability.py",
        "--license",
        f"{DEVICES['iqm']}:1",
        "--timeout",
        "30",
    )
    unavailable = json.loads((cluster.RUNTIME / "jobs" / "iqm.json").read_text())
    for definition in unavailable["qdmi"]["devices"]:
        definition["enabled"] = False
    (cluster.RUNTIME / "jobs" / "iqm-offline.json").write_text(json.dumps(unavailable))
    failed = cluster.controller(
        *environment("iqm"),
        "MQT_CORE_QDMI_CONFIG_FILE=/jobs/iqm-offline.json",
        *monitor,
        check=False,
        timeout=45,
    )
    assert failed.returncode == 1
    assert cluster.license_record(DEVICES["iqm"])["Reserved"] == "1"
    waiting = submit("iqm", probe=False)
    (cluster.RUNTIME / "jobs" / f"release-{iqm}").touch()
    cluster.wait_for("the first IQM job to finish", lambda: cluster.job_finished(iqm))
    cluster.wait_for(
        "the next IQM job to wait without a compute node",
        lambda: cluster.job_matches(waiting, "PENDING", node="", reason="Licenses"),
    )
    assert cluster.job_matches(braket, "RUNNING")
    assert not (cluster.RUNTIME / "jobs" / f"iqm-{waiting}.json").exists()
    cluster.job(
        *environment("braket"), "mqt-core-qdmi-check", "--device", DEVICES["braket"], "--timeout", "30", timeout=45
    )
    cluster.controller(*environment("iqm"), *monitor, timeout=45)
    cluster.wait_for_result("iqm", waiting, "the pending IQM job to execute after recovery")
    cluster.wait_for("the recovered IQM allocation to finish", lambda: cluster.job_finished(waiting))
    (cluster.RUNTIME / "jobs" / f"release-{braket}").touch()
    cluster.wait_for("the Braket job to finish", lambda: cluster.job_finished(braket))
    for device in DEVICES.values():
        cluster.assert_license(device, total=1, used=0, free=1)
        assert cluster.license_record(device)["Reserved"] == "0"
    cluster.LOGGER.info("IQM and Braket ran together; an IQM outage blocked only IQM admission.")


def main() -> None:
    """Run the combined wheel deployment using existing provider source checkouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--iqm", required=True, type=Path)
    parser.add_argument("--braket", required=True, type=Path)
    options = parser.parse_args()
    for name in ("IQM_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(name):
            parser.error(f"{name} must be set for the credentialed workload")
    for vendor in DEVICES:
        wheels = list((options.dist / vendor).glob("*.whl"))
        source = getattr(options, vendor).resolve()
        if len(wheels) != 1 or not (source / "test/slurm/probe.py").is_file():
            parser.error(f"Supply one {vendor} wheel under --dist/{vendor} and its source checkout")
        cluster.COMPOSE_ENV[f"MQT_CORE_SLURM_{vendor.upper()}"] = str(source)
    cluster.main(
        ("--dist", str(options.dist), "--compose-file", str(Path(__file__).with_suffix(".yml"))),
        workload=test_vendors,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
