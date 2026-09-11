---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Exchange structured programs with jeff

The previous tutorials chose an execution format and a device. Sometimes another
compiler needs to work on the program first. What should cross that boundary?
How can you check that serialization preserved the program, including its
control flow and phase?

[jeff](https://github.com/unitaryfoundation/jeff) provides a common exchange
format for quantum compilers. This tutorial uses MQT Core on both sides of a
handoff, including a fresh Python process. Another compiler can occupy either
side when it supports the format version and operations being exchanged.

The notebook runs independently with the
[tutorial setup](index.md#run-the-notebooks). MQT Core includes the required
jeff integration. The {doc}`jeff guide <../jeff>` describes the API and
conversion limits.

{download}`Download this notebook <../_build/jupyter_execute/tutorials/jeff_exchange.ipynb>`.

```{code-cell} ipython3
:tags: [hide-input]
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from IPython.display import Code, display

from mqt.core.mlir import (
    JeffProgram,
    OutputFormat,
    build_functionality,
    compile_program,
    sample,
    submit_program,
)
```

## Start with a program whose full unitary matters

Prepare a Bell state and include a global phase of $\pi/4$. A global phase is
invisible to this circuit's measurement counts, but a full-unitary comparison
can detect whether the compiler preserved it.

```{code-cell} ipython3
source = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
gphase(pi / 4);
h q[0];
cx q[0], q[1];
"""
original_unitary = build_functionality(source)
print(original_unitary)
```

Two qubits give a $4\times4$ matrix, so comparing the full unitary is cheap
here. Dense matrices grow exponentially; larger compiler tests need an oracle
suited to their program and scale.

## Convert to an exchange representation

Compile the program to jeff and inspect its MLIR form:

```{code-cell} ipython3
exchange = compile_program(source, output=OutputFormat.JEFF)
display(Code(exchange.ir, language="mlir"))
```

The operations represent allocation, gates, controls, and the global phase. The
module metadata identifies its entry point, format version, and producing tool.
This is a representation a compiler can work on before selecting a target.

`exchange.ir` is useful for inspection. To produce the binary exchange payload,
serialize the program:

```{code-cell} ipython3
payload = exchange.to_bytes()
assert isinstance(payload, bytes)
print(f"Serialized payload: {len(payload)} bytes")
```

The payload is self-contained. Sending an in-memory MLIR object would require
sharing its compiler context; serialized bytes or a file form the handoff.

## Receive, compile, and check semantics

Deserialize the payload and enter the receiving compiler's optimization
pipeline. Predict whether the new MLIR text must look identical for the program
to be equivalent.

```{code-cell} ipython3
received = JeffProgram.from_bytes(payload)
optimized = compile_program(received, output=OutputFormat.QCO_OPTIMIZED)
received_unitary = build_functionality(optimized)
np.testing.assert_allclose(received_unitary, original_unitary, rtol=0, atol=1e-12)
print("Largest matrix-entry difference:", np.max(np.abs(received_unitary - original_unitary)))
```

:::{dropdown} Explain the result
SSA names, operation order, and intermediate representations may change. The
comparison checks the resulting unitary, including phase and logical wire order,
rather than requiring a particular printed representation.
:::

Remove the phase from the source and compare again:

```{code-cell} ipython3
without_phase = build_functionality(source.replace("gphase(pi / 4);", ""))
np.testing.assert_allclose(received_unitary, np.exp(1j * np.pi / 4) * without_phase, rtol=0, atol=1e-12)
assert not np.allclose(received_unitary, without_phase)
print("The global phase survived the exchange.")
```

**Experiment:** change the phase in the source to `pi / 2`, rerun the handoff,
and update the expected phase factor above. The full matrix changes while the
Bell measurement probabilities stay the same.

## Hand a file to an independent process

Write a `.jeff` file and start a fresh Python interpreter to load, compile, and
sample it. The receiving process has no access to the producer's MLIR context.
The temporary directory only keeps the example from leaving files behind.

```{code-cell} ipython3
:tags: [hide-input]
consumer = """
import json
import sys
from pathlib import Path
from mqt.core.mlir import OutputFormat, compile_program, sample

program = compile_program(Path(sys.argv[1]), output=OutputFormat.QCO_OPTIMIZED)
print(json.dumps(sample(program, shots=64, seed=17)))
"""
```

```{code-cell} ipython3
with TemporaryDirectory() as directory:
    path = Path(directory) / "exchange.jeff"
    exchange.write(path)
    completed = subprocess.run(
        [sys.executable, "-c", consumer, str(path)], check=True, capture_output=True, text=True
    )

received_counts = json.loads(completed.stdout)
assert set(received_counts) <= {"00", "11"}
assert sum(received_counts.values()) == 64
received_counts
```

The receiver can also use `JeffProgram.from_file(path)` when it needs the jeff
program object before choosing a compiler output. This example validates MQT's
producer and consumer paths; an integration with another compiler must also
check that compiler's supported operations and format version.

## Exchange a measurement-dependent loop

A flat list of gates cannot capture the following program's meaning by itself.
Its loop condition depends on a measurement. As in the QIR tutorial, the program
flips a measured `1` to `0` and then exits.

```{code-cell} ipython3
feedback_source = """OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
bit result = measure q;
while (result) {
    x q;
    result = measure q;
}
"""
feedback_exchange = compile_program(feedback_source, output=OutputFormat.JEFF)
feedback_received = JeffProgram.from_bytes(feedback_exchange.to_bytes())
restored_qco = compile_program(feedback_received, output=OutputFormat.QCO)
assert sample(restored_qco, shots=64, seed=17) == {"0": 64}
```

Inspect the loop on each side of the conversion:

```{code-cell} ipython3
:tags: [hide-input]
for name, program, operation in (
    ("jeff", feedback_received, "jeff.while"),
    ("QCO", restored_qco, "scf.while"),
):
    assert operation in program.ir
    print(name)
    print("\n".join(line for line in program.ir.splitlines() if operation in line))
```

The loop remains explicit. Its state carries the qubit and classical information
between iterations. The receiver can continue structured compilation instead of
receiving only a predetermined gate sequence.

## Choose an execution format after exchange

The received jeff program is still a compiler input. Compile it for DDSIM, which
selects Adaptive QIR, then submit the resulting payload through QDMI:

```{code-cell} ipython3
from mqt.core.qdmi.driver import open_device

device = open_device("mqt.ddsim.default")
compiled = compile_program(feedback_received, target=device)
job = submit_program(compiled, target=device, num_shots=64, custom1=17)
assert job.wait()
assert job.get_counts() == {"0": 64}
print("Execution format:", compiled.program_format.name)
print("Counts:", job.get_counts())
```

This completes the path from a structured exchange payload to a device result.
Use jeff before physical mapping: MQT's conversion does not preserve static
hardware site IDs. The {doc}`jeff guide <../jeff>` lists further constraints on
arrays, helper functions, and custom operations.

For the next experiment, exchange one of the supported structured
{doc}`benchmarks <../benchmarks>`, then compare its evaluated results before and
after the handoff. Use the
{doc}`compiler guide <../mlir/mqt_compiler_collection>` to choose another output
format or optimization pipeline.
