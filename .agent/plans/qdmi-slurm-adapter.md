# Add a QDMI adapter for Slurm licenses

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date. Maintain this document in accordance with `.agent/PLANS.md`.

## Purpose / Big Picture

A Slurm job can use one local license environment value to open the registered
QDMI device with the same stable ID. The public names identify the mechanism:
`fomac::slurm::openDeviceFromLicense()` in C++ and
`mqt.core.qdmi.slurm.open_device_from_license()` in Python. The QDMI driver
remains independent of Slurm.

The adapter accepts one local unit license value, opens a fresh device session
from the persistent definition, and checks the device state. It does not verify
the allocation, authorize access, inject credentials, or apply per-job QDMI
configuration.

## Progress

- [x] (2026-08-13 03:38Z) Reconstruct the branch on the QDMI namespace layer.
- [x] (2026-08-13 03:38Z) Add the C++ Slurm adapter and Python submodule.
- [x] (2026-08-13 03:38Z) Move grammar and status tests into the FoMaC test
      component.
- [x] (2026-08-13 04:23Z) Generate recursive stubs and run the complete FoMaC,
      DDSIM status, QDMI driver, Python, lint, documentation, wheel, and
      installed-wheel smoke checks.
- [x] (2026-08-13 12:31Z) Rebuild the layer on the finalized QDMI namespace head
      and move the Python test into the QDMI test tree.
- [x] (2026-08-13 12:45Z) Complete an independent adversarial review and address
      the changed-surface clang-tidy and coverage findings.
- [x] (2026-08-13 14:40Z) Define the environment variable as selection metadata,
      not as allocation evidence or authorization.

## Surprises & Discoveries

- Observation: The adapter needs no Slurm client library. Slurm exports the
  license expression in `SLURM_JOB_LICENSES` before the job runs. The process
  can modify this environment variable.
- Observation: The existing test QDMI provider can report configured device
  states. This keeps the unavailable-state test provider-independent and out of
  the QDMI driver tests.
- Observation: DDSIM reported `OFFLINE` before its first job even though a fresh
  session could submit work. The adapter correctly rejected that state. DDSIM
  now reports `IDLE` before its first job, changes to `BUSY` while work runs,
  and returns to `IDLE` when the work completes.

## Decision Log

- Decision: Put the adapter in `fomac::slurm` and `mqt.core.qdmi.slurm`.
  Rationale: The names state the mechanism and do not imply support for other
  resource managers. Date/Author: 2026-08-13, Lukas Burgholzer and Codex.
- Decision: Accept exactly one license with an implicit count or count one.
  Rationale: One job needs one device handle. A larger configured pool admits
  more independent jobs, not more handles in one job. Date/Author: 2026-08-13,
  Lukas Burgholzer and Codex.
- Decision: Keep credential handling in each provider. Rationale: IQM, Amazon
  Braket, and future providers use different standard credential sources. Slurm
  admission does not own authentication. Date/Author: 2026-08-13, Lukas
  Burgholzer and Codex.
- Decision: Treat `SLURM_JOB_LICENSES` as selection metadata only. Rationale:
  the process can modify the environment and can call the public device-opening
  API directly. Provider credentials or operating-system isolation must enforce
  access. Date/Author: 2026-08-13, Lukas Burgholzer and Codex.

## Outcomes & Retrospective

The focused C++ grammar, device-state, and DDSIM transition tests pass. The
complete FoMaC and QDMI driver suites pass. The QDMI Python suite passes on
Python 3.10 and 3.14. Recursive stub generation includes the
`mqt.core.qdmi.slurm` module. Full lint and the documentation build pass. The
independent review found no remaining local blocker. Platform CI remains.

## Context and Orientation

`include/mqt-core/fomac` and `src/fomac` contain owning C++ wrappers above the
QDMI client interface. `bindings/qdmi` exposes those wrappers through the native
`mqt.core.qdmi` extension. `src/qdmi/driver` owns the device registry and must
not contain scheduler-specific parsing.

Slurm places the license expression in `SLURM_JOB_LICENSES`. A local static
license has the form `name` or `name:count`. Remote licenses add `@server`.
Commas and pipes represent compound expressions. This adapter supports only the
single-device subset. The value is not proof of an allocation.

## Plan of Work

Add `fomac/Slurm.hpp` and `src/fomac/Slurm.cpp`. Parse an owned environment
string. Use explicitly typed character pointers for `std::from_chars` so the
code works with the Microsoft standard library. Compare the parsed stable ID to
the registered IDs, then call the existing fresh-open operation without
overrides. Query the QDMI device status and accept `IDLE` or `BUSY`.

Expose the function in a sibling Python `slurm` submodule. Keep QDMI entities in
`mqt.core.qdmi` and ordinary registration and opening functions in
`mqt.core.qdmi.driver`. Add C++ tests for the complete grammar and status
contract. Add a Python binding test with the DDSIM device. Document the boundary
between Slurm admission, environment-based selection, device status, provider
queues, and provider authorization.

## Validation and Acceptance

The C++ tests cover an absent or empty variable, whitespace, malformed counts,
signed counts, zero, non-unit counts, trailing characters, integer overflow,
remote licenses, unknown IDs, AND expressions, OR expressions, implicit and
explicit unit counts, `IDLE`, `BUSY`, and every other QDMI device state. Each
device-state rejection diagnostic contains the stable ID and reported status.

Stub generation must add `python/mqt/core/qdmi/slurm.pyi`. Python tests must
open DDSIM through the binding and reject compound and non-unit expressions. The
complete lint and documentation sessions must pass. Platform CI provides the
Windows compiler and free-threaded Python gates.

## Idempotence and Recovery

Tests restore the original process environment after each case. Device
registrations use stable test IDs and insert only when absent. If the lower
namespace layer changes, rebuild this branch on its new exact head and rerun all
affected checks before publication.

## Interfaces and Dependencies

The final public interfaces are:

    #include "fomac/Slurm.hpp"
    auto device = fomac::slurm::openDeviceFromLicense();

    from mqt.core.qdmi import slurm
    device = slurm.open_device_from_license()

The adapter depends on MQT Core FoMaC and the QDMI registry. It does not link to
Slurm or a provider SDK.
