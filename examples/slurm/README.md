# Update Slurm device availability

`update_availability.py` uses Slurm 25.11+ license-only reservations to keep
jobs pending while a QDMI device is unavailable. It reserves the device's full
local license count before each check and removes that reservation only after a
successful check. Running jobs continue, and unrelated licenses remain usable.
No SPANK plugin or SlurmDBD is required.

Run one monitor per device, as root or the configured `SlurmUser`, on an
administrative host with Slurm clients and the site's QDMI checker, catalogue,
device implementation, and credentials. The controller itself needs no device
implementation. The monitor's credentials must represent site access; a user's
failed authentication must not change cluster-wide availability. Jobs retain
their own credentials and must handle failures after allocation.

For `Licenses=mqt.ddsim.default:2`, run:

```console
python3 update_availability.py --license mqt.ddsim.default:2 --checker /opt/qdmi/bin/mqt-core-qdmi-check
```

The count must equal the full configured capacity. The monitor owns
`qdmi-unavailable-mqt.ddsim.default`; reserve this name for it, and do not
create overlapping license reservations. Each invocation renews the block for
one year. Use `--block-only` to close admission without probing. Stop both the
timer and its service before using this option for a manual outage, so an
in-flight or later healthy check cannot reopen it.

Exit status is `0` for a successful readiness check or `--block-only`, `1` for a
failed or timed-out probe with admission blocked, and `2` for invalid arguments
or a controller operation failure. Alert on controller failures: an unreachable
controller cannot be updated. Checker diagnostics are suppressed because they
can contain credentials. Inspect device failures separately under the site's
logging policy.

The checker accepts `IDLE` and `BUSY` as operational states. The license count
is the site's concurrency policy; this check does not measure the remote queue
or reserve capacity at the service. Probe failures, including expired monitor
credentials, leave admission closed until a later successful probe.

## Poll with systemd

Copy the example to `/opt/mqt-core/examples/slurm/`. Adapt this service for each
device as `/etc/systemd/system/qdmi-availability.service`:

```ini
[Unit]
Description=Update QDMI device admission in Slurm
Wants=network-online.target
After=network-online.target munge.service

[Service]
Type=oneshot
User=slurm
Environment=MQT_CORE_QDMI_CONFIG_FILE=/etc/mqt-core/site.qdmi.json
ExecStart=/usr/bin/python3 /opt/mqt-core/examples/slurm/update_availability.py --license mqt.ddsim.default:2 --checker /opt/qdmi/bin/mqt-core-qdmi-check
TimeoutStartSec=45
```

Provide any device-specific credential-file reference through a root-owned
service configuration; do not put tokens or keys in command-line arguments.
Install `/etc/systemd/system/qdmi-availability.timer`:

```ini
[Unit]
Description=Poll QDMI device readiness

[Timer]
OnBootSec=1s
OnUnitInactiveSec=30s
AccuracySec=1s

[Install]
WantedBy=timers.target
```

Before opening the queue, establish blocks for every monitored license. For an
existing cluster, this sequence pauses new allocations while retaining running
jobs. Apply it to every partition that can request these licenses:

```console
scontrol update PartitionName=compute State=DOWN
python3 /opt/mqt-core/examples/slurm/update_availability.py --license mqt.ddsim.default:2 --block-only
systemctl daemon-reload
systemctl enable --now qdmi-availability.timer
scontrol update PartitionName=compute State=UP
```

For a new controller, start the affected partitions with `State=DOWN` in
`slurm.conf`, then establish the blocks before setting them `UP`. Incorporate
this ordering into site startup procedures.

Polling gives a readiness snapshot. A device can fail after a successful probe,
and a stopped timer after success leaves the license open. Supervise the timer
and service; if the site requires a maximum observation age, an independent
watchdog must run `--block-only` when that age is exceeded. A service
`OnFailure` handler can alert or close admission, but cannot detect a stopped
timer by itself. Inspect both `Free` and `Reserved` in `scontrol show licenses`:
reserved tokens can still appear in `Free`.
