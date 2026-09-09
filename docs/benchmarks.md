---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Structured quantum benchmarks

MQT Core defines each structured quantum benchmark by benchmark-specific
parameters and an analytic reference. A benchmark instance can produce a
structured QC program, a resolved manifest, and a stable case ID. The generated
program returns one classical register named `result`. Outcome strings are
big-endian: the highest-index result bit is the leftmost character.

## Discover the catalog

The command-line registry is the current list of available families. Each family
has its own instance specification schema. These cells execute the CLI and fail
if it exits unsuccessfully; JSON formatting only makes its output easier to
read.

```{code-cell} ipython3
import json
import subprocess

catalog = subprocess.run(["mqt-core-bench", "list"], check=True, capture_output=True, text=True)
print(json.dumps(json.loads(catalog.stdout), indent=2))
```

```{code-cell} ipython3
description = subprocess.run(["mqt-core-bench", "describe", "qft"], check=True, capture_output=True, text=True)
print(json.dumps(json.loads(description.stdout), indent=2))
```

## Configure a typed instance

Python exposes each benchmark through a family-specific type. Parameterized
families also expose option types. The QFT input below is the uniform
superposition of multiples of two. Both circuit methods use the same logical
output and reference.

```{code-cell} ipython3
from mqt.core.bench import qft


benchmark = qft.QFT(
    qft.Options(
        qubits=3,
        period_exponent=1,
        method=qft.Method.SEMICLASSICAL,
    )
)
print("Method:", benchmark.options.method)
print("Output:", benchmark.output.name)
print("Width:", benchmark.output.width)
```

Each family validates its instance when it creates one. Fixed families need no
options.

## Inspect the canonical instance specification and manifest

A canonical instance specification records every resolved default. A manifest
also binds the logical output, reference descriptor, family-definition version,
and case ID.

```{code-cell} ipython3
import json


instance_specification = json.loads(benchmark.instance_specification_json)
manifest = json.loads(benchmark.manifest_json)
print("Instance specification:")
print(json.dumps(instance_specification, indent=2))
print("\nManifest summary:")
print(
    json.dumps(
        {
            "case_id": manifest["case_id"],
            "outputs": manifest["outputs"],
            "reference": manifest["reference"],
        },
        indent=2,
    )
)
```

## Query and evaluate the reference

For three output bits and period exponent one, QFT has two equal peaks.

```{code-cell} ipython3
probabilities = {
    outcome: benchmark.probability(outcome) for outcome in ("000", "100", "010")
}
assert probabilities == {"000": 0.5, "100": 0.5, "010": 0.0}
print(json.dumps(probabilities, indent=2))
```

```{code-cell} ipython3
evaluation = benchmark.evaluate({"000": 500, "100": 500})
print(
    json.dumps(
        {
            "total_variation_distance": evaluation.total_variation_distance,
            "squared_hellinger_fidelity": evaluation.squared_hellinger_fidelity,
            "success_probability": evaluation.success_probability,
        },
        indent=2,
    )
)
```

Total variation distance zero and squared Hellinger fidelity one identify an
exact distribution. Some benchmark families also report a success probability
for a distinguished success outcome.

## Generate structured IR

Generation returns a {py:class}`~mqt.core.mlir.QCProgram`, the program type used
by the [MQT Core MLIR compiler collection](mlir/python_compiler_collection.md).
The program can enter the normal compiler pipeline.

```{code-cell} ipython3
program = benchmark.generate()
assert program.is_valid
structured_ir = program.ir
assert program.copy().to_qco().is_valid
print(structured_ir)
```

## Run the command-line workflow

The CLI writes the program first and its manifest last. A manifest is therefore
the completion marker. Existing output files always cause an error.

```{code-cell} ipython3
:tags: [hide-input]

import tempfile
from pathlib import Path


temporary = tempfile.TemporaryDirectory()
root = Path(temporary.name)
instance_specification_path = root / "instance-specification.json"
counts_path = root / "counts.json"
output_directory = root / "generated"
```

```{code-cell} ipython3
:tags: [remove-output]

instance_specification_path.write_text(
    benchmark.instance_specification_json, encoding="utf-8"
)
counts_path.write_text(
    json.dumps({"schema_version": 1, "counts": {"000": 5, "100": 5}}),
    encoding="utf-8",
)
```

```{code-cell} ipython3
generation = subprocess.run(
    ["mqt-core-bench", "generate", "--instance-specification", str(instance_specification_path),
     "--format", "qc", "--output", str(output_directory)],
    check=True, capture_output=True, text=True,
)
generated = json.loads(generation.stdout)
print("Generated", generated["benchmark"], "as", generated["format"])
```

```{code-cell} ipython3
manifest_path = next(output_directory.glob("*.manifest.json"))
program_path = next(output_directory.glob("*.qc.mlir"))
print("Program:", program_path.name)
print("Manifest:", manifest_path.name)
```

```{code-cell} ipython3
evaluation_result = subprocess.run(
    ["mqt-core-bench", "evaluate", "--manifest", str(manifest_path), "--counts", str(counts_path)],
    check=True, capture_output=True, text=True,
)
metrics = json.loads(evaluation_result.stdout)["metrics"]
assert metrics["total_variation_distance"] == 0
print(json.dumps(metrics, indent=2))
```

```{code-cell} ipython3
:tags: [remove-cell]

temporary.cleanup()
```

Use `--format jeff` instead of `--format qc` to write a binary `jeff` program.
The output format changes the file name, but not the semantic case ID.

## C++ API

The installed `MQT::CoreBench` target provides typed parameters, references,
evaluation, instances, instance specifications, and manifests.

```cpp
#include "bench/Grover.hpp"

#include <cassert>

int main() {
  const mqt::bench::Grover benchmark{{.markedBitstring = "101"}};
  const auto evaluation = benchmark.evaluate({{"101", 1000}});
  assert(evaluation.successProbability == 1.0);
}
```

```cmake
find_package(mqt-core CONFIG REQUIRED)
target_link_libraries(my-benchmark PRIVATE MQT::CoreBench)
```

The source build also provides `MQT::CoreBenchGenerate`. It exposes typed
`mqt::bench::generate(...)` overloads from `mlir/bench/Generate.h` and returns a
`mlir::QCProgram`. This target is not installed until MQT Core installs the
wider MLIR compiler API.

## Add a benchmark

Adding a family requires five extension points:

1. Add one `(TYPE, STEM, ID, DEFINITION_VERSION)` row to
   `include/mqt-core/bench/BenchmarkFamilies.inc`. Its expansions provide the
   public JSON declarations and the synchronized semantic and MLIR registry
   glue.
2. Add the typed instance, any options and validation, an analytic reference,
   and evaluation under `include/mqt-core/bench/` and `src/bench/`. Add the
   family-specific parameter JSON, reference JSON, parser, and schema body to
   `src/bench/JSON.cpp`.
3. Declare and implement the structured emitter under `mlir/bench/`, add its
   source to the program library, and declare the typed `generate(...)`
   overload. The catalog supplies the generation wrapper and JSON dispatch row.
4. Add the explicit Python types in a family registration source under
   `bindings/bench/`, register its direct submodule in `register_bench.cpp`, and
   add the source to `bindings/bench/CMakeLists.txt`.
5. Test the reference, strict instance specification JSON, emitter structure,
   `jeff` conversion, and Python generation.

`BenchmarkFamilies.inc` is the sole family catalog. Do not add a private family
list, a generic option map, or a public base class.

## Reproducibility contract

Instance specifications reject unknown fields and invalid values. The case ID
does not depend on a path or output format. Parsing a manifest checks its
resolved parameters, logical output, reference, definition version, and case ID.
Before evaluation, normalize backend results to the manifest's big-endian
`result` order.

## Benchmark families

### QFT addition

The `qft-adder` family adds two equal-width operands. `REGISTER` stores the
addend in qubits and applies controlled phases; `CONSTANT` combines the known
addend into one phase per accumulator qubit. Both use the same exact QFT and
inverse QFT. `WRAP` keeps an n-bit sum, while `CARRY` keeps an extra sum bit.

```{code-cell} ipython3
from mqt.core import mlir
from mqt.core.bench import qft_adder

adder = qft_adder.QFTAdder(
    qft_adder.Options(
        addend="110",
        accumulator="011",
        method=qft_adder.Method.CONSTANT,
        overflow=qft_adder.Overflow.CARRY,
    )
)
assert mlir.sample(adder.generate(), shots=128, seed=17) == {"1001": 128}
```

Operands are big-endian strings; leading zeros set their common width. The
accumulator and constant addends must be binary. Register addends may also use
`+` for independently prepared $|+\rangle$ qubits, such as `addend="1+0"`.
Register results concatenate the addend and sum so their correlation remains
observable. Constant results contain only the sum. `expected_result` is the
unique logical outcome for basis inputs and `None` for a superposed addend. The
total sum width, including an optional carry bit, is limited to 1024.

### Modular multiplier

The `modular-multiplier` family uses the controlled modular arithmetic circuit
from Figures 5 and 6 of
[Beauregard's circuit for Shor's algorithm](https://arxiv.org/abs/quant-ph/0205095).
It computes `control || multiplicand || product`, with
`product = control * multiplier * multiplicand mod modulus`. The product
register starts at zero and retains its leading overflow bit; a work qubit must
return to zero. This is an out-of-place multiplier.

The classical `multiplier` and canonical `modulus` are equal-width binary
strings with $0 < \mathtt{multiplier} < \mathtt{modulus}$. The required
`multiplicand` has the same width and accepts `0`, `1`, and `+`, as in the QFT
adder. A `+` prepares an independent $|+\rangle$ qubit. The `control` accepts
`"0"`, `"1"`, or `"+"`, and defaults to `"1"`. Widths range from 2 to 63 bits.

```python
from mqt.core.bench import modular_multiplier

benchmark = modular_multiplier.ModularMultiplier(
    modular_multiplier.Options(multiplier="011", modulus="101", multiplicand="111")
)
assert benchmark.expected_result == "11110001"  # control=1, input=7, product=1
assert benchmark.evaluate({"11110001": 100}).success_probability == 1.0
```

Basis inputs have one exact `expected_result`, so TVD and success probability
provide a direct check independent of width. An all-zero output fails for this
nonzero example. Test different inputs and both control values to exercise
wraparound and the inactive path.

For superposed inputs, `expected_result` is `None`. The reference assigns
probability $2^{-k}$ to each allowed input and its correct product, where $k$ is
the number of `+` input bits, including the control. `success_probability` is
the shot-weighted fraction matching both the configured inputs and the
arithmetic relation. With $S$ shots, empirical TVD is at least
$\max(0,1-S/2^k)$, even for ideal execution. Keep $k$ small for sampling-based
distribution checks at large widths.

Computational-basis measurements cannot detect arbitrary relative-phase errors.
Native tests therefore also compare complete coherent states and require clean
work-qubit recovery. A single correct basis result does not certify a unitary on
every input.
