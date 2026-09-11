# Add a QDMI adapter for Slurm licenses

Status: historical implementation record.

## Goal and scope

A Slurm job can use one local license environment value to open the registered
QDMI device with the same stable ID. The public names identify the mechanism:
`fomac::slurm::openDeviceFromLicense()` in C++ and
`mqt.core.qdmi.slurm.open_device_from_license()` in Python. The QDMI driver
remains independent of Slurm.

The adapter accepts one local unit license value, opens a fresh device session
from the persistent definition, and checks the device state. It does not verify
the allocation, authorize access, inject credentials, or apply per-job QDMI
configuration.

## Constraints

- The adapter needs no Slurm client library. Slurm exports the license
  expression in `SLURM_JOB_LICENSES` before the job runs. The process can modify
  this environment variable.

- The existing test QDMI provider can report configured device states. This
  keeps the unavailable-state test provider-independent and out of the QDMI
  driver tests.

- DDSIM reported `OFFLINE` before its first job even though a fresh session
  could submit work. The adapter correctly rejected that state. DDSIM now
  reports `IDLE` before its first job, changes to `BUSY` while work runs, and
  returns to `IDLE` when the work completes.

## Decisions

- Put the adapter in `fomac::slurm` and `mqt.core.qdmi.slurm`. Rationale: The
  names state the mechanism and do not imply support for other resource
  managers.

- Accept exactly one license with an implicit count or count one. Rationale: One
  job needs one device handle. A larger configured pool admits more independent
  jobs, not more handles in one job.

- Keep credential handling in each provider. Rationale: IQM, Amazon Braket, and
  future providers use different standard credential sources. Slurm admission
  does not own authentication.

- Treat `SLURM_JOB_LICENSES` as selection metadata only. Rationale: the process
  can modify the environment and can call the public device-opening API
  directly. Provider credentials or operating-system isolation must enforce
  access.

## Outcome and validation

The focused C++ grammar, device-state, and DDSIM transition tests pass. The
complete FoMaC and QDMI driver suites pass. The QDMI Python suite passes on
Python 3.10 and 3.14. Recursive stub generation includes the
`mqt.core.qdmi.slurm` module. Full lint and the documentation build pass. Final
hosted CI was not recorded.

## Code and ownership

`include/mqt-core/fomac` and `src/fomac` contain owning C++ wrappers above the
QDMI client interface. `bindings/qdmi` exposes those wrappers through the native
`mqt.core.qdmi` extension. `src/qdmi/driver` owns the device registry and must
not contain scheduler-specific parsing.

Slurm places the license expression in `SLURM_JOB_LICENSES`. A local static
license has the form `name` or `name:count`. Remote licenses add `@server`.
Commas and pipes represent compound expressions. This adapter supports only the
single-device subset. The value is not proof of an allocation.

## Acceptance

The C++ tests cover an absent or empty variable, whitespace, malformed counts,
signed counts, zero, non-unit counts, trailing characters, integer overflow,
remote licenses, unknown IDs, AND expressions, OR expressions, implicit and
explicit unit counts, `IDLE`, `BUSY`, and every other QDMI device state. Each
device-state rejection diagnostic contains the stable ID and reported status.

Stub generation must add `python/mqt/core/qdmi/slurm.pyi`. Python tests must
open DDSIM through the binding and reject compound and non-unit expressions. The
complete lint and documentation sessions must pass. Platform CI provides the
Windows compiler and free-threaded Python gates.

## Interfaces

The final public interfaces are:

    #include "fomac/Slurm.hpp"
    auto device = fomac::slurm::openDeviceFromLicense();

    from mqt.core.qdmi import slurm
    device = slurm.open_device_from_license()

The adapter depends on MQT Core FoMaC and the QDMI registry. It does not link to
Slurm or a provider SDK.
