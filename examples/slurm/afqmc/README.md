# QC-AFQMC through Slurm HRES and QDMI

This prototype runs the full SC26 H₂/STO-3G example: prepare 4,096
matchgate-shadow circuits, compile and execute 1,024 shots per circuit, then
propagate 128 walkers per trial for 240 imaginary-time steps. Three dependent
Slurm jobs release their allocations between stages. The quantum job holds one
device HRES unit and opens its own QDMI session. Compilation targets that same
device.

DDSIM, QDMI-on-IQM, and the Amazon Braket QDMI device execute real programs. The
latter two connect to local HTTP services: IQM native PRX/CZ programs run
through Qiskit's statevector; Braket OpenQASM runs through Amazon's local
simulator. The fixtures deliberately reorder measurement fields to exercise
reconstruction. They model this four-qubit gate subset, not cloud validation,
latency, billing, calibration, or hardware noise. No cloud credentials or jobs
are needed.

## Run

Use a rootful Linux Docker engine with cgroup v2. See the
[playground guide](https://mqt.readthedocs.io/projects/core/en/latest/qdmi/slurm.html)
for host requirements and the HRES/credential boundary. The full example needs a
32 GiB quantum allocation; allow at least 48 GiB physical RAM for one workflow.
All containers share the host. Small smoke runs use the same allocation limits.

Run from the Core repository root in a fresh runtime directory:

```bash
sh examples/slurm/afqmc/prepare.sh
export MQT_CORE_SLURM_RUNTIME="$PWD/build/slurm-afqmc"
export MQT_CORE_SLURM_CORE="$PWD"
export MQT_CORE_SLURM_WORKLOAD="$MQT_CORE_SLURM_RUNTIME/upstream"
export MQT_CORE_SLURM_PACKAGES="$(cat examples/slurm/afqmc/requirements.txt)"
docker compose -p afqmc -f examples/slurm/compose.yml \
  -f examples/slurm/afqmc/compose.yml up --build -d --wait
docker compose -p afqmc -f examples/slurm/compose.yml \
  -f examples/slurm/afqmc/compose.yml exec --user 10000:10000 login bash --login
```

In that login shell:

```bash
scontrol show licenses
python /workspace/examples/slurm/afqmc/submit.py --device iqm.emerald.mock
# Alternative local implementations:
python /workspace/examples/slurm/afqmc/submit.py --device amazon.braket.sv1
python /workspace/examples/slurm/afqmc/submit.py --device mqt.ddsim.default
squeue --me
sacct -X --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES%100
```

The submitter prints the three job IDs and a unique `/jobs/afqmc-*` directory.
Use `--snapshots 8 --shots 64 --walkers 4 --steps 8` for a smoke run. Failed
predecessors cancel their dependents. Each allocation has a one-hour limit;
provider execution waits are bounded to 30 minutes.

Preparation and propagation use classical nodes; the quantum stage uses one
four-CPU, 32 GiB QAN. Propagation uses four MPI ranks on four nodes. Different
provider simulations sample independently, so energy estimates need not match
bit for bit. They are ideal simulations, not measurements of hardware fidelity.

## Validate and inspect

Run these commands in the login shell. Replace the artifact directory with the
one printed by your completed run:

```bash
for device in mqt.ddsim.default iqm.emerald.mock amazon.braket.sv1; do
  srun --partition=quantum --licenses="$device" --cpus-per-task=4 --mem=4096M \
    --time=00:05:00 python /workspace/test/slurm/check_afqmc.py --device "$device"
done
python /workspace/test/slurm/check_afqmc.py --directory /jobs/afqmc-REPLACE
```

The readout check sends every four-bit basis state with reversed classical
output destinations. Artifact checks cover hashes, shot counts, four distinct
rank nodes, chronological stage boundaries, and scientific invariants. Retained
files include source QASM, QDMI input payloads, raw logical shots, compact
shadows, Slurm identity/timing, Hamiltonian checks, force-bias checks, walker
histories, and energy curves. `sacct` supplies independent scheduler accounting.

For local protocol inspection, `submission.json` retains provider job IDs. The
IQM fixture exposes its received request at `/api/v1/jobs/<uuid>/payload`; the
Braket fixture exposes it at `/quantum-task/<uuid>/payload`. Fetch these from
`http://iqm:8080` or `http://braket:8080` inside the cluster before restarting
the services. They contain JSON request bodies, never authentication headers.
The Braket native provider prepares terminal measurements, so its service
request is distinct from the original QDMI input returned by `get_program`.

The scientific path validates every Majorana transformation, vacuum-preserving
trial preparation, measured-overlap contraction, force-bias derivative, and
exact-eigenstate local energy. The six-determinant cache is exact for this
active space. Reported error bars cover independent walkers conditional on one
shadow set; they exclude shadow uncertainty, time-step/projection error, and
phaseless bias. No quantum advantage or multi-host scaling claim is made.

The services retain results in memory until restarted. Their `/availability`
endpoint controls local device health for failure tests. They have no published
host ports and use only dummy AWS fixture credentials. Never expose them as a
production service.

## Connect a live service

Use the base playground with the provider packages and appropriate HRES records;
omit the local-service overlay. Configure the actual catalogue endpoint/device
ID and mount user credentials on login/QAN hosts and separate probe credentials
on the controller. For Braket, configure `AMZN_BRAKET_TASK_RESULTS_S3_URI` and
the usual AWS credential chain; remove all local endpoint overrides. IQM uses
its provider's token-file setup. Request the exact selected ID with
`--licenses`.

The IQM compiler emits device-native JSON from discovered gates and topology.
The Braket path uses the provider's existing Qiskit serializer and currently
requires dense placement on the first four simulator sites. It is a simulator
example, not an implementation of physical-QPU placement or verbatim submission.
Remote reservations, queue guarantees, hardware calibration and paid execution
need separate validation. None was exercised by the local prototype.

## Sources

- Slurm: pinned development revision and archive digest in the parent
  Dockerfile, plus `slurm-hres-reconfigure.patch` to restore dynamic QAN
  membership on controller reconfiguration. Lua HRES and dynamic membership are
  subject to change before release.
- Core: the checkout built into the image; Python 3.14 throughout.
- Providers and scientific packages: exact versions in `requirements.txt`.
  Braket is pinned to the native classical-output correction in
  [draft PR #288](https://github.com/munich-quantum-software/amazon-braket-qdmi-device/pull/288);
  the example needs that fix for asymmetric readout.
- Amazon's
  [Quantum Monte Carlo Chemistry example](https://github.com/amazon-braket/amazon-braket-examples/tree/16cd791da7c3ec7e104851eb3bc502d00be1c1c3/examples/hybrid_quantum_algorithms/Quantum_Monte_Carlo_Chemistry):
  the preparation script downloads that revision and checks the archive digest;
  `source-hashes.json` checks every imported scientific source. Upstream code is
  used under its Apache-2.0 license and remains unmodified.
- MQSC's SC26 presentation prototype supplies the H₂ case, trial preparation,
  convention checks, and exact active-space contractions. This example includes
  the scientific workflow without presentation assets or captured results.

Stop only this project with the same Compose files and `down`; use
`down --volumes` only to discard its accounting history. The runtime contains
generated keys/passwords and is excluded from Git. Preserve selected result
directories separately from credentials when sharing evidence.
