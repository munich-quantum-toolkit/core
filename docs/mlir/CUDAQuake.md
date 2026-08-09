---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# CUDA-Q Quake interoperability

MQT Core can parse CUDA-Q's textual, reference-semantics Quake representation
and translate it directly to the {doc}`QC` dialect. It can also emit the same
conservative Quake form from QC. CUDA-Q is not an MQT Core dependency: the
boundary between the two packages is one textual MLIR serialization.

The live CUDA-Q path in this notebook is used when `cudaq` is installed. The
regular documentation build instead executes the checked-in CUDA-Q 0.15
fixture, so the page remains reproducible without CUDA-Q.

```{code-cell} ipython3
from pathlib import Path

from mqt.core.mlir import QCProgram, QIRProfile, QuakeProgram

try:
    import cudaq
except ImportError:
    cudaq = None

if cudaq is not None:
    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mz(q)

    quake_source = str(cudaq.synthesize(bell))
    source_kind = "live cudaq.synthesize"
else:
    candidates = (
        Path("fixtures/cudaq-0.15-bell.qke"),
        Path("docs/mlir/fixtures/cudaq-0.15-bell.qke"),
    )
    fixture = next(path for path in candidates if path.exists())
    quake_source = fixture.read_text(encoding="utf-8")
    source_kind = "checked-in CUDA-Q 0.15 fixture"

print(source_kind)
print(quake_source)
```

CUDA-Q kernels with runtime arguments must be specialized before import. Pass
the arguments to `cudaq.synthesize` and serialize the returned module:

```python
quake_source = str(cudaq.synthesize(kernel, *arguments))
```

## Quake to QC, QCO, QIR, and DDSIM

Parse Quake and translate it to QC. The source `QuakeProgram` is consumed by
default; `copy=True` is available on the Python conversion when both objects
are needed.

```{code-cell} ipython3
quake = QuakeProgram.from_mlir_str(quake_source)
qc = quake.to_qc()
print(qc.ir)
```

The normal QC to QCO path remains the only optimization path. There is no
separate Quake to QCO API.

```{code-cell} ipython3
qco = qc.to_qco()
qco.cleanup()
qco.merge_single_qubit_rotation_gates()
optimized_qc = qco.to_qc()

qir = optimized_qc.to_qir(QIRProfile.BASE, copy=True)
assert "__quantum__qis__h__body" in qir.llvm_ir
print(qir.llvm_ir)
```

The bundled DDSIM QDMI device accepts QIR Base Profile bitcode, which lets the
same converted program be executed without CUDA-Q.

```{code-cell} ipython3
from mqt.core.fomac import ProgramFormat, open_device

device = open_device("mqt.ddsim.default")
job = device.submit_job(
    qir.llvm_ir,
    ProgramFormat.QIR_BASE_STRING,
    num_shots=128,
)
job.wait()
counts = job.get_counts()
counts
```

## QC to Quake and CUDA-Q

QC export produces an argument-free CUDA-Q kernel with the requested entry
point. A nonzero global phase is rejected because CUDA-Q 0.15 reference-form
Quake cannot represent it exactly. Set `ignore_global_phase=True` only when
dropping that phase is intentional.

```{code-cell} ipython3
emitted = optimized_qc.to_quake(name="mqt_bell", copy=True)
print(emitted.ir)
```

When CUDA-Q is installed, `merge_quake_source` turns the emitted text into a
live kernel decorator, which can be sampled on CUDA-Q's selected target. This
also works with external targets such as a QDMI-backed CUDA-Q device.

```{code-cell} ipython3
if cudaq is not None:
    @cudaq.kernel
    def merge_anchor():
        pass

    cudaq_kernel = merge_anchor.merge_quake_source(emitted.ir)
    cudaq_counts = cudaq.sample(cudaq_kernel, shots_count=1024)
    print(cudaq_counts)
else:
    print("Install CUDA-Q to execute the reverse-direction example.")
```

## Supported subset and compatibility

| Area | Behavior |
| --- | --- |
| Common kernels | Static `ref`/`veq` allocation and access, standard gates and parameters, adjoints, ordered positive/negative controls, reset, named measurements, scalar measurement feedback, named kernel application, `cc.if`, and bounded `cc.loop` are supported. |
| Runtime arguments | Specialize first with `str(cudaq.synthesize(kernel, *arguments))`. |
| Quantum semantics | Import requires reference-form `ref`/`veq`; SSI `wire`/`cable` programs are rejected. |
| Unsupported operations | State initialization, noise, custom-unitary definitions, unspecialized dynamic allocations/accesses, indirect callables, and unsupported CC forms fail at the first relevant operation or type. |
| Global phase | QC to Quake rejects a nonzero phase by default. `ignore_global_phase=True` explicitly drops it. `quake.phase` will be added after it appears in a CUDA-Q release. |
| Version policy | CUDA-Q 0.15 is the initial textual baseline. Compatibility is maintained per syntax feature and retained fixtures, not as a promised CUDA-Q version range. |
| Dependencies | CUDA-Q is installed separately. MQT Core never links CUDA-Q's MLIR libraries. |

The compatibility dialect deliberately models only this surface. If a future
CUDA-Q release changes relevant syntax, MQT Core can add a focused parser or
operation adjustment while keeping older fixtures supported where practical.
