---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Compile and execute Shor's algorithm

Factor 21 with a structured quantum program, the compiler, and the bundled DDSIM
simulator. Install MQT Core in a Python 3.11 or newer environment as described
in {doc}`installation`. The Python wheel includes the compiler and simulator
used here.

## Factor a number

{term}`Shor's algorithm` combines quantum {term}`order finding` with classical
factor recovery. For a chosen base $a$ coprime to $N$, the quantum circuit
estimates phases of the modular multiplication $x \mapsto ax \bmod N$. Continued
fractions turn measured phases into candidate exponents; modular exponentiation
and greatest common divisors verify possible factors.

The following example is the same source as the README. The factoring driver
selects bases and recovers factors. Its `run` callback owns device selection,
compilation, shot count, and execution seed.

```{code-cell} ipython3
:load: _build/readme_example.py
```

Compilation and submission are separate operations:
{py:func}`~mqt.core.mlir.compile_program` produces a reusable
{py:class}`~mqt.core.mlir.CompiledProgram`, and
{py:func}`~mqt.core.mlir.submit_program` creates a device job. `job.wait()`
waits for completion and reports execution failure. Both Adaptive QIR and
OpenQASM 3 execute the same circuit; `custom1=17` selects DDSIM's random seed.
Custom job properties are device-specific.

The driver first checks for even numbers, primes, and perfect powers. These need
no quantum execution and use zero attempts. Otherwise it tries base 2, then
seeded random bases, up to `max_attempts=16`. A base with a nontrivial common
divisor can reveal a factor before execution. A circuit whose counts yield no
factors causes another attempt. The result reports `SUCCESS`, `PRIME`, or
`ATTEMPTS_EXHAUSTED`, an optional sorted factor pair, and the number of
attempted bases. It returns one pair, not a recursive prime factorization.

## Inspect one order-finding circuit

Use a single-base benchmark to inspect the quantum and classical steps. For
$N=21$ and $a=2$, the circuit uses $2n+3=13$ qubits, where
$n=\operatorname{bit\_width}(21)=5$. One query qubit is measured and reset for
each of the $2n=10$ phase bits. The value register persists across rounds;
modular arithmetic restores its temporary qubits after each multiplication.
Earlier measurements control phase corrections on the query qubit.

```{code-cell} ipython3
benchmark = shor.Shor(shor.Options(number=21, base=2))
counts = run(benchmark)
evaluation = benchmark.evaluate(counts)
assert sum(counts.values()) == 64
assert evaluation.factors == (3, 7)
print(f"Verified factors: {evaluation.factors}")
print(f"Successful shot fraction: {evaluation.success_probability:.3f}")
```

Counts put the highest-index phase bit on the left, so each bitstring encodes an
integer $y$ and an estimate $y/2^{10}$. Equivalent OpenQASM and QIR payloads use
the same bit order. The benchmark accepts `job.get_counts()` directly.

Select an observed outcome that yields factors and inspect its rational
approximation:

```{code-cell} ipython3
from fractions import Fraction
from math import gcd

outcome = next(
    bits for bits in sorted(counts, key=lambda bits: counts[bits], reverse=True)
    if benchmark.evaluate({bits: 1}).factors is not None
)
phase = Fraction(int(outcome, 2), 2**benchmark.output.width)
approximation = phase.limit_denominator(benchmark.options.number - 1)
r = approximation.denominator
assert r % 2 == 0
assert pow(benchmark.options.base, r, benchmark.options.number) == 1
half_power = pow(benchmark.options.base, r // 2, benchmark.options.number)
p = gcd(half_power - 1, benchmark.options.number)
q = gcd(half_power + 1, benchmark.options.number)
assert tuple(sorted((p, q))) == (3, 7)
print(f"Measured {outcome}: {phase}, near {approximation}")
print(f"Candidate exponent: {r}; gcd recovery: {p}, {q}")
```

For example, $171/1024$ is close to $1/6$. The candidate exponent 6 satisfies
$2^6 \equiv 1 \pmod{21}$, and $2^{6/2}=8$ gives $\gcd(8-1,21)=7$ and
$\gcd(8+1,21)=3$.

This single approximation illustrates a successful sample. The native
{py:meth}`~mqt.core.bench.shor.Shor.evaluate` examines exact
{term}`continued-fraction convergents <continued fraction>` with denominators
below $N$ and verifies each candidate. A reduced phase such as $2/6=1/3$ need
not retain the order in its denominator. Zero phases, unsuitable exponents, and
trivial greatest common divisors are ordinary unsuccessful samples. More shots
and, when needed, new bases give further chances to recover factors.

`success_probability` is the observed fraction of shots that independently yield
a verified pair. It is not a distance from an ideal distribution. Shor provides
a verification reference and does not report total variation distance or
Hellinger fidelity.

## Continue

- Read the [Shor benchmark reference](benchmarks.md#shor-order-finding) for
  input limits and JSON interfaces.
- Try the [QPE examples](benchmarks.md#quantum-phase-estimation) to isolate
  phase estimation with a known eigenstate.
- Work through the {doc}`compiler tutorial <tutorials/index>` to explain
  representations, optimizations, control flow, and hardware constraints.
- Use {doc}`mlir/target_compilation` and {doc}`qdmi/driver` to compile for other
  devices and inspect their capabilities.
