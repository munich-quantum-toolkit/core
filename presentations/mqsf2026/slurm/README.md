# Capture H2 AFQMC on Slurm

This workload uses the current
[Slurm playground](../../../examples/slurm/README.md): one quantum access node
runs the PennyLane/QDMI shadow batch with a DDSIM license; an `afterok`
dependency then releases four classical ranks on four compute nodes. The nodes
are Docker containers on one physical host. No hardware jobs run.

The image reuses the playground's services, admission checks, partitions,
accounting, and shared filesystem. It installs pinned Python 3.14 chemistry
dependencies and an existing native Core build, avoiding another compiler build
inside Docker. The availability monitor's image-local shebang selects that same
Python environment. Its logic and the repository's playground configuration are
unchanged.

## Build the image

Use a Linux native build for the Docker engine's architecture, linked against
system libraries compatible with Ubuntu 26.04. On the DGX Spark, build and stage
the optimized preset:

```bash
cmake --preset release-clang-ipo
cmake --build --preset release-clang-ipo
cmake --install build/release-clang-ipo --prefix "$PWD/build/mqsf-slurm/native"
mkdir -p build/mqsf-slurm/metadata
cp -a .venv/lib/python*/site-packages/mqt_core-*.dist-info build/mqsf-slurm/metadata/
docker build \
  --build-context core=. \
  --build-context cluster=examples/slurm \
  --build-context native=build/mqsf-slurm/native \
  --build-context metadata=build/mqsf-slurm/metadata \
  -t mqsf-afqmc-slurm:latest presentations/mqsf2026/slurm
```

The metadata comes from the existing editable Core installation and supplies
PennyLane's plugin registration. The native stage and checkout supply the actual
implementation. Record their revision and binary hashes with the capture; the
editable distribution's version string alone does not identify those binaries.

## Run the workflow

Obtain the pinned AWS example; the recorder checks every imported source hash:

```bash
git clone https://github.com/amazon-braket/amazon-braket-examples.git build/mqsf-slurm/aws
git -C build/mqsf-slurm/aws checkout 16cd791da7c3ec7e104851eb3bc502d00be1c1c3
export MQT_CORE_SLURM_CORE="$PWD"
export MQT_CORE_SLURM_WORKLOAD="$PWD/build/mqsf-slurm/aws/examples/hybrid_quantum_algorithms/Quantum_Monte_Carlo_Chemistry"
export MQT_CORE_SLURM_RUNTIME="$PWD/build/mqsf-slurm/runtime"
sh examples/slurm/prepare.sh "$MQT_CORE_SLURM_RUNTIME"
docker compose -p mqsf-afqmc \
  -f examples/slurm/compose.yml -f presentations/mqsf2026/slurm/compose.yml \
  up -d --no-build --wait
.venv/bin/python presentations/mqsf2026/capture_slurm_afqmc.py \
  --runtime "$MQT_CORE_SLURM_RUNTIME" \
  --compose-file presentations/mqsf2026/slurm/compose.yml
```

The fixed defaults are H2/STO-3G at 1.6 Å, a paired trial with excitation angle
0.6 radians, 4,096 shadow circuits with 1,024 shots each, 128 independent
walkers for each trial, 240 steps of 0.02 Ha⁻¹, and auxiliary-field/permutation
seed 17. DDSIM uses its default random shot stream, whose exact ordered samples
are saved. The correlated and Hartree–Fock trials use the same walker seeds. No
run or seed is selected based on the final energy curves.

An initial smoke run can add `--snapshots 8 --shots 64 --walkers 4 --steps 8`.
Both runs retain native batching, payload identity, bit order, molecular
Hamiltonian, overlap, force-bias, and local-energy checks. The runner rejects
failed allocations or a result that does not contain four ranks on four nodes.

Results are under `$MQT_CORE_SLURM_RUNTIME/jobs/`:

- `quantum.json`: actual native programs, ordered shots, and adapter/status
  observations.
- `afqmc-source.qasm`: snapshot 0's actual submitted OpenQASM 3 bytes.
- `afqmc.json`: complete science and per-rank walker observations.
- `afqmc-slurm.json`: the complete capture with Slurm accounting and observed
  scheduler transitions.
- `program-completions.tsv`: DDSIM's actual per-program terminal status,
  monotonic timestamp, executor budget, and assigned worker slots.
- `slurm-JOBID.out`: allocation logs.

Before submission, four compiler workers invoke the installed `mqt-cc` on each
actual adapter-generated OpenQASM payload, targeting `mqt.ddsim.default` and
producing QIR 2.1 Base Profile. One multi-program QDMI job then executes those
unchanged QIR payloads with four DDSIM executor slots. Preparation, individual
compiler calls, native completions, and result retrievals retain their real
timestamps. The original OpenQASM and submitted QIR both retain source hashes.

The native `MQT_MQSF_PROGRAM_TRACE` hook is opt-in and requires one isolated job
per output path. Its rows are emitted when native results become available;
result retrieval remains a separate observed adapter event. This hook does not
switch DDSIM to serial per-shot simulation.

Only observed states are emitted; a fast transition can be absent. Native device
completion and application completion are distinct because result extraction
happens between them. Worker timestamps share the Docker host's monotonic clock;
this clock assumption does not extend to a cluster of separate physical hosts.
Energy error bars cover walker variation conditional on this single measured
shadow set, excluding shadow sampling error and systematic bias.

After retaining the evidence, stop this task's cluster:

```bash
docker compose -p mqsf-afqmc \
  -f examples/slurm/compose.yml -f presentations/mqsf2026/slurm/compose.yml \
  down --volumes
```

The runtime directory includes private cluster keys and passwords; package the
completed JSON, source, and relevant logs only.
