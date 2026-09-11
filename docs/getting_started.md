---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Compile and execute a quantum program

This walkthrough uses quantum phase estimation (QPE) to show the path from a
structured program to a device result. Install MQT Core in a Python 3.11 or
newer environment as described in {doc}`installation`. The Python wheel includes
the compiler and the DDSIM simulator used here.

## Estimate a known phase

QPE estimates a phase $\phi$ from an eigenvalue $e^{2\pi i\phi}$ of a unitary
operation. The benchmark supplies a phase gate and its known eigenstate, so we
can compare execution against an analytic reference.

The following example is the same source as the README. It estimates $3/8$ with
eight bits of precision. Compilation and submission are separate operations:
{py:func}`~mqt.core.mlir.compile_program` produces a reusable
{py:class}`~mqt.core.mlir.CompiledProgram`, and
{py:func}`~mqt.core.mlir.submit_program` creates a device job.

```{code-cell} ipython3
:load: _build/readme_example.py
```

Eight result bits encode an integer $k$ and the estimate $k/2^8$. Here,
`01100000` is 96, so the estimate is $96/256=3/8$.

DDSIM selects Adaptive QIR for this program. Its QDMI shots and counts place the
highest-index output bit on the left, so the benchmark evaluates
`job.get_counts()` directly. Equivalent OpenQASM and QIR payloads use the same
bitstring order.

`job.wait()` waits for completion and reports execution failure. The counts
contain the final phase register; measurements used internally for feedback are
not separate benchmark outputs.

## Standard versus iterative QPE

Both methods estimate the same phase. Standard QPE uses a register of phase
qubits and an inverse quantum Fourier transform. Iterative QPE reuses one phase
qubit, measures and resets it after each step, and uses earlier measurements to
choose later corrections.

| Method    |                      Qubits at eight-bit precision | Execution requirement                        |
| --------- | -------------------------------------------------: | -------------------------------------------- |
| Standard  |     9: eight phase qubits and one eigenstate qubit | Coherent phase register and terminal readout |
| Iterative | 2: one reused phase qubit and one eigenstate qubit | Mid-circuit measurement, reset, and feedback |

These counts describe this phase-gate benchmark before target optimization. They
do not include the larger registers required by other controlled unitaries.

```{code-cell} ipython3
standard = qpe.QPE(
    qpe.Options(precision=8, phase=Fraction(3, 8), method=qpe.Method.STANDARD)
)
standard_program = compile_program(standard.generate(), target=device)
standard_job = submit_program(standard_program, target=device, num_shots=1024)
standard_job.wait()
standard_counts = standard_job.get_counts()
assert standard_counts == counts
assert standard.evaluate(standard_counts).total_variation_distance < 1e-12
print(f"Standard QPE: {standard_counts}")
print(f"Iterative QPE: {counts}")
```

QPE also appears in Shor's order-finding procedure, where the controlled unitary
performs modular multiplication. This example uses a phase gate with a supplied
eigenstate. It demonstrates phase estimation and feedback, not modular
arithmetic, order finding, or factoring.

## A phase between representable values

The phase $1/3$ is not an integer multiple of $1/256$. QPE therefore produces a
distribution over nearby estimates. More shots characterize that distribution;
increasing the number of phase bits makes the estimation grid finer.

```{code-cell} ipython3
approximate = qpe.QPE(
    qpe.Options(precision=8, phase=Fraction(1, 3), method=qpe.Method.ITERATIVE)
)
approximate_program = compile_program(approximate.generate(), target=device)
approximate_job = submit_program(
    approximate_program, target=device, num_shots=4096, custom1=17
)
approximate_job.wait()
approximate_counts = approximate_job.get_counts()
evaluation = approximate.evaluate(approximate_counts)
assert evaluation.total_variation_distance < 0.08

total = sum(approximate_counts.values())
print("Estimate   Observed   Ideal")
for bits in sorted(approximate_counts, key=lambda bits: approximate_counts[bits], reverse=True)[:4]:
    estimate = Fraction(int(bits, 2), 2**approximate.output.width)
    print(f"{str(estimate):9}  {approximate_counts[bits] / total:.3f}      {approximate.probability(bits):.3f}")
print(f"Total variation distance: {evaluation.total_variation_distance:.3f}")
```

The total variation distance compares the complete sampled distribution with the
analytic reference. Zero means agreement; finite-shot samples generally have a
nonzero distance. `custom1=17` selects DDSIM's random seed for this example;
custom job properties are device-specific.

## Continue

- Work through the {doc}`compiler tutorial <tutorials/index>` to explain
  representations, optimizations, control flow, and hardware constraints.
- Try the [repeat-until-success benchmark](benchmarks.md#repeat-until-success)
  for a measurement-controlled retry loop.
- Read {doc}`mlir/mqt_compiler_collection` for input formats, program objects,
  transformations, and export.
- Use {doc}`mlir/target_compilation` and {doc}`qdmi/driver` to compile for other
  devices and inspect their capabilities.
- Explore {doc}`benchmarks` for other program families and reference metrics.
