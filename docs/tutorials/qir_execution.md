---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Execute programs with QIR

The hardware tutorial produced a device-compatible program. What is inside that
payload, and how does it express measurement feedback? Here you will follow a
program into QIR, compare the Base and Adaptive profiles, and connect QIR output
records to ordinary measurement counts.

This notebook runs independently with the
[tutorial setup](index.md#run-the-notebooks). Execution uses local DDSIM. The
{doc}`QIR guide <../qir/index>` provides the API and runtime reference.

{download}`Download this notebook <../_build/jupyter_execute/tutorials/qir_execution.ipynb>`.

```{code-cell} ipython3
:tags: [hide-input]
from IPython.display import Code, display
from qiskit import QuantumCircuit
from qiskit.visualization import plot_distribution

from mqt.core.mlir import OutputFormat, QIRProfile, compile_program
from mqt.core.qdmi import CustomProperty, ProgramFormat
from mqt.core.qdmi.driver import open_device

device = open_device("mqt.ddsim.default")
shots = 1024
seed = 17
```

## Predict a static circuit

Start with the Bell circuit from the first tutorial. Both qubits are measured at
the end, so the Base Profile can express it. Predict its possible bitstrings
before running the circuit:

```{code-cell} ipython3
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])
circuit.draw("mpl")
```

A single call compiles that Qiskit circuit to QIR. LLVM text lets us inspect the
calls and metadata; bitcode serializes the same program for execution.

```{code-cell} ipython3
base = compile_program(circuit, output=OutputFormat.QIR_BASE)
assert base.profile is QIRProfile.BASE
display(Code(base.llvm_ir, language="llvm"))
```

Look for three parts of the output:

- `__quantum__qis__...` calls apply gates and measurements.
- `__quantum__rt__...record_output` calls select which results leave the
  program.
- The entry-point attributes and module flags declare the profile, QIR version,
  and resource requirements.

The `i64` returned by the entry point is its exit code. Measurement results
leave through output-recording calls, not through that return value.

```{code-cell} ipython3
bitcode = base.to_bitcode()
job = device.submit_job(bitcode, ProgramFormat.QIR_BASE_MODULE, shots, custom1=seed)
assert job.wait()
counts = job.get_counts()
assert set(counts) == {"00", "11"}
assert sum(counts.values()) == shots
assert abs(counts["00"] / shots - 0.5) < 0.08
plot_distribution(counts, title="Base Profile: Bell measurements")
```

Try submitting `base.llvm_ir` with `ProgramFormat.QIR_BASE_STRING` instead. Does
changing the serialization change the circuit's meaning?

```{code-cell} ipython3
text_job = device.submit_job(base.llvm_ir, ProgramFormat.QIR_BASE_STRING, shots, custom1=seed)
assert text_job.wait()
assert text_job.get_shots() == job.get_shots()
print(f"LLVM text: {len(base.llvm_ir.encode())} bytes; bitcode: {len(bitcode)} bytes")
```

The equality here uses the same DDSIM execution path and seed. Different
runtimes or sampling algorithms need not produce the same random sequence.

## Let a measurement control another qubit

Replace the controlled-X gate with a classical decision: measure the first
qubit, then flip the second if that result was `1`. Will this change the joint
output distribution? Will it preserve an entangled state?

```{code-cell} ipython3
feedback_source = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
bit flag = measure q[0];
if (flag) {
    x q[1];
}
bit answer = measure q[1];
"""

adaptive = compile_program(feedback_source, output=OutputFormat.QIR_ADAPTIVE)
adaptive_job = device.submit_job(
    adaptive.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, shots, custom1=seed
)
assert adaptive_job.wait()
feedback_counts = adaptive_job.get_counts()
assert set(feedback_counts) == {"00", "11"}
assert sum(feedback_counts.values()) == shots
plot_distribution([counts, feedback_counts], legend=["Bell circuit", "Measurement feedback"])
```

:::{dropdown} Explain the result
Both programs have perfectly correlated outputs in this measurement basis. The
feedback program first measures and collapses the first qubit, then uses a
classical bit to prepare the second. Matching counts in one basis do not prove
that the programs preserve the same quantum state or implement the same unitary.
:::

Inspect the Adaptive program's result read and conditional branch:

```{code-cell} ipython3
:tags: [hide-input]
for line in adaptive.llvm_ir.splitlines():
    if "read_result" in line or "br i1" in line or '"qir_profiles"' in line:
        print(line)
assert adaptive.profile is QIRProfile.ADAPTIVE
```

**Experiment:** change `x q[1]` to `z q[1]`. Qubit 1 starts in $|0\rangle$;
predict whether the classical outputs remain correlated.

```{code-cell} ipython3
variant = compile_program(feedback_source.replace("x q[1]", "z q[1]"), output=OutputFormat.QIR_ADAPTIVE)
variant_job = device.submit_job(variant.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, shots, custom1=seed)
assert variant_job.wait()
assert set(variant_job.get_counts()) == {"00", "01"}
plot_distribution(variant_job.get_counts(), title="Z leaves the answer qubit in zero")
```

QDMI places the last output bit on the left: these strings read `answer, flag`.
The `answer` bit is always zero, while `flag` can be zero or one.

## Keep a feedback loop in the program

The Adaptive Profile also supports loops when the required capabilities are
available. This loop flips and remeasures a qubit only if its first measurement
was `1`. It exits after at most one iteration and returns `0`.

```{code-cell} ipython3
loop_source = """OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
bit result = measure q;
while (result) {
    x q;
    result = measure q;
}
"""
loop = compile_program(loop_source, output=OutputFormat.QIR_ADAPTIVE)
```

The compiler derives the module flags from the lowered instructions. Inspect how
this program declares its conditional loop and resource management:

```{code-cell} ipython3
:tags: [hide-input]
for line in loop.llvm_ir.splitlines():
    if line.startswith("!"):
        print(line)
assert '"backwards_branching"' in loop.llvm_ir
```

`backwards_branching` describes the loop capability; LLVM prints the
conditional-loop bit pattern as `i2 -2`. The dynamic resource flags and `arrays`
describe this program's qubit and result storage. These flags describe the
compiled program's requirements, not a request to enable a feature on hardware.

Try compiling the loop for a Base-only payload. The target compiler rejects the
unsupported control flow:

```{code-cell} ipython3
try:
    compile_program(loop_source, target=device, program_format=ProgramFormat.QIR_BASE_MODULE)
except ValueError as error:
    print(error)
else:
    raise AssertionError("A conditional loop requires an Adaptive payload")
```

## Read the records behind the counts

Enable DDSIM's QIR output capture for two shots. Each shot may execute several
measurements, but only the final `result` is recorded as program output.

```{code-cell} ipython3
recorded = device.submit_job(
    loop.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=2, custom1=seed, custom2=True
)
assert recorded.wait()
output = recorded.get_custom_result(CustomProperty.CUSTOM1, str)
assert isinstance(output, str)
assert output.count("START\n") == 2 and output.count("END\t0\n") == 2
assert recorded.get_counts() == {"0": 2}
print(output, end="")
print("Counts:", recorded.get_counts())
```

The header identifies the output schema. `START` and `END` delimit each shot;
`OUTPUT` lines preserve result labels and typed values. The histogram contains
recorded measurement bits, not every intermediate measurement performed in the
loop.

Capture runs once per shot and retains the text in memory. Leave it disabled for
ordinary sampling; it also prevents the terminal-sampling statevector from being
retained. The {doc}`QIR guide <../qir/index>` covers both output schemas,
resource lifetimes, and state-extraction restrictions.

Continue with {doc}`qdmi_execution` to explore device discovery, payload
selection, repeated jobs, and the different kinds of results behind this API.
