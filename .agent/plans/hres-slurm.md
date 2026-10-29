# HRES Slurm playground

Status: implemented and locally validated; draft review remains pending.

## Scope and decisions

Replace the license-reservation monitor with Slurm HRES admission in
`examples/slurm/`. HRES owns local capacity, QAN placement, and availability;
applications own QDMI sessions, credentials, compilation, and remote jobs.

Use two explicitly named QANs (`qan1`, `qan2`) and declarative HRES membership.
Classical nodes remain scalable. Dynamic QAN discovery and a custom Slurm
reconfiguration patch are unnecessary under this limit. Build unmodified Slurm
26.11.0rc1 revision `ba178ef837c4ba799fb3de72bcdaf30bc2b1616d` with Lua support.

The Lua callback admits IDLE/BUSY devices and disables admission on errors. Its
separate Python probe has a fixed ten-second timeout and two-second kill grace.
Running jobs retain allocations. The status probe uses site credentials; user
jobs authenticate independently. A Slurm allocation does not reserve remote
capacity or establish user authorization.

Keep one Python 3.14 environment and the external-provider integration runner
interface used by provider repositories. The AFQMC implementation, scientific
dependencies, local provider services, and recordings belong in the separate
private presentations repository. Core documents generic dependent Slurm jobs.

## Validation

The 13 host tests in `test/python/test_slurm_integration.py`, Lua 5.4 callback
self-check, Compose configuration validation, and `uvx nox -s lint` pass.

A fresh cluster with two static QANs and two dynamic classical nodes passes
capacity, placement, outage/recovery, running allocations across
reconfiguration, independent job environments, and persistent accounting. The
run took 327 seconds including image build/startup and 80 seconds for execution
checks. Executable documentation and rendered-guide inspection pass.

No cloud or hardware jobs were submitted. Hosted CI has not been monitored.
Slurm HRES remains a pinned development interface; remote reservation and
site-specific credentials are outside this playground's validation.
