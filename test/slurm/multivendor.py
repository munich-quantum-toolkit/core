# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exercise DDSIM, IQM, and Braket independently on one credentialed Slurm cluster."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
from pathlib import Path

import run_integration as cluster

DEVICES = {"ddsim": "mqt.ddsim.default", "iqm": "iqm.emerald.mock", "braket": "amazon.braket.sv1"}
AWS_CREDENTIALS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def environment(vendor: str) -> tuple[str, ...]:
    """Keep one shared catalogue while submitting only the needed credentials."""
    unrelated = (
        *(AWS_CREDENTIALS if vendor != "braket" else ()),
        *(("IQM_TOKEN", "IQM_TOKENS_FILE") if vendor != "iqm" else ()),
    )
    return (
        "env",
        *(argument for name in unrelated for argument in ("-u", name)),
        "MQT_CORE_QDMI_CONFIG_FILE=/jobs/devices.json",
    )


def submit(vendor: str, *, hold: bool = True) -> str:
    """Submit one licensed batch job, holding the allocation when requested."""
    unrelated = AWS_CREDENTIALS if vendor == "iqm" else ("IQM_TOKEN", "IQM_TOKENS_FILE")
    metadata = (
        "import json, os, socket; from pathlib import Path; "
        f"assert os.environ['SLURM_JOB_LICENSES'] in ('{DEVICES[vendor]}', '{DEVICES[vendor]}:1'); "
        f"assert not any(os.environ.get(key) for key in {unrelated!r}); "
        f"Path('/jobs/{vendor}-' + os.environ['SLURM_JOB_ID'] + '.json').write_text("
        "json.dumps({'node': socket.gethostname(), 'licenses': os.environ['SLURM_JOB_LICENSES']}))"
    )
    body = "set -eu\n"
    body += f"python3 -c {shlex.quote(metadata)}\n"
    if hold:
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
    licenses = ",".join(f"{device}:1" for vendor, device in DEVICES.items() if vendor != "ddsim")
    configuration.write_text(configuration.read_text().replace("Licenses=", f"Licenses={licenses},", 1))
    cluster.controller("scontrol", "reconfigure")
    definitions = []
    for vendor in ("iqm", "braket"):
        cluster.controller(
            "env", "PROVIDER_INSTALL_MODE=wheel", "sh", f"/{vendor}/test/slurm/setup.sh", f"/jobs/{vendor}.json"
        )
        definitions.extend(json.loads((cluster.RUNTIME / "jobs" / f"{vendor}.json").read_text())["qdmi"]["devices"])
    catalogue = cluster.RUNTIME / "jobs" / "devices.json"
    catalogue.write_text(json.dumps({"schema-version": 1, "qdmi": {"devices": definitions}}))
    (cluster.RUNTIME / "qdmi.env").write_text("MQT_CORE_QDMI_CONFIG_FILE=/jobs/devices.json\n")
    cluster.controller("systemctl", "restart", "qdmi-admission.service")
    for device in DEVICES.values():
        cluster.wait_for(f"{device} admission", lambda device=device: cluster.license_record(device)["Reserved"] == "0")
    cluster.job(
        "env",
        "MQT_CORE_QDMI_CONFIG_FILE=/jobs/devices.json",
        "srun",
        "--immediate=5",
        "--time=5",
        f"--licenses={','.join(DEVICES.values())}",
        "python3",
        "/workspace/test/slurm/multivendor_job.py",
        timeout=300,
    )

    iqm, braket = (submit(vendor) for vendor in ("iqm", "braket"))
    for vendor, job_id in (("iqm", iqm), ("braket", braket)):
        cluster.wait_for_result(vendor, job_id, f"the {vendor} allocation")
    for vendor, job_id in (("iqm", iqm), ("braket", braket)):
        assert cluster.job_matches(job_id, "RUNNING")
        cluster.assert_license(DEVICES[vendor], total=1, used=1, free=0)
        assert cluster.load_result(vendor, job_id)["node"] in cluster.NODES

    service = f"qdmi-availability@{DEVICES['iqm']}:1.service"
    cluster.controller("systemctl", "stop", service)
    offline = json.loads(catalogue.read_text())
    for definition in offline["qdmi"]["devices"]:
        if definition["id"] == DEVICES["iqm"]:
            definition["session"]["base-url"] = "http://127.0.0.1:1"
    catalogue.write_text(json.dumps(offline))
    cluster.controller("systemctl", "start", service)
    cluster.wait_for(
        "the failed IQM probe",
        lambda: (
            cluster.controller("systemctl", "show", service, "--property=SubState", "--value").stdout.strip()
            == "auto-restart"
        ),
    )
    assert cluster.license_record(DEVICES["iqm"])["Reserved"] == "1"
    waiting = submit("iqm", hold=False)
    (cluster.RUNTIME / "jobs" / f"release-{iqm}").touch()
    cluster.wait_for("the first IQM job to finish", lambda: cluster.job_finished(iqm))
    cluster.wait_for(
        "the next IQM job to wait without a compute node",
        lambda: cluster.job_matches(waiting, "PENDING", node="", reason="Licenses"),
    )
    assert cluster.job_matches(braket, "RUNNING")
    assert not (cluster.RUNTIME / "jobs" / f"iqm-{waiting}.json").exists()
    for vendor in ("ddsim", "braket"):
        cluster.job(*environment(vendor), "mqt-core-qdmi-check", "--device", DEVICES[vendor], timeout=45)
    catalogue.write_text(json.dumps({"schema-version": 1, "qdmi": {"devices": definitions}}))
    cluster.controller("systemctl", "restart", service)
    cluster.wait_for_result("iqm", waiting, "the pending IQM job to start after recovery")
    cluster.wait_for("the recovered IQM allocation to finish", lambda: cluster.job_finished(waiting))
    (cluster.RUNTIME / "jobs" / f"release-{braket}").touch()
    cluster.wait_for("the Braket job to finish", lambda: cluster.job_finished(braket))
    for vendor, device in DEVICES.items():
        capacity = 2 if vendor == "ddsim" else 1
        cluster.assert_license(device, total=capacity, used=0, free=capacity)
        cluster.wait_for(
            f"{device} admission after recovery",
            lambda device=device: cluster.license_record(device)["Reserved"] == "0",
        )
    cluster.LOGGER.info("DDSIM, IQM, and Braket ran together; an IQM outage blocked only IQM admission.")


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
    for vendor in ("iqm", "braket"):
        wheels = list((options.dist / vendor).glob("*.whl"))
        source = getattr(options, vendor).resolve()
        if len(wheels) != 1 or not (source / "test/slurm/setup.sh").is_file():
            parser.error(f"Supply one {vendor} wheel under --dist/{vendor} and its source checkout")
        cluster.COMPOSE_ENV[f"MQT_CORE_SLURM_{vendor.upper()}"] = str(source)
    cluster.main(
        ("--dist", str(options.dist), "--compose-file", str(Path(__file__).with_suffix(".yml"))),
        workload=test_vendors,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
