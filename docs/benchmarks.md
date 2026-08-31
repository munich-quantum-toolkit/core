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
has its own instance specification schema.

```{code-cell} ipython3
!mqt-core-bench list
```

```{code-cell} ipython3
!mqt-core-bench describe qft
```

## Configure a typed instance

Python exposes benchmark-specific option types. The QFT input below is the
uniform superposition of multiples of two. Both circuit methods use the same
logical output and reference.

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

Each benchmark family validates its options when it creates an instance.

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
:tags: [remove-cell]

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
!mqt-core-bench generate --instance-specification {instance_specification_path} --format qc --output {output_directory}
```

```{code-cell} ipython3
manifest_path = next(output_directory.glob("*.manifest.json"))
program_path = next(output_directory.glob("*.qc.mlir"))
print("Program:", program_path.name)
print("Manifest:", manifest_path.name)
```

```{code-cell} ipython3
!mqt-core-bench evaluate --manifest {manifest_path} --counts {counts_path}
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

Adding a family requires five explicit extension points:

1. Add typed options, validation, an analytic reference, and evaluation under
   `include/mqt-core/bench/` and `src/bench/`.
2. Add its instance specification schema and evaluation callback to the private
   semantic registry in `src/bench/JSON.cpp`.
3. Add one structured emitter and one generation callback to the private MLIR
   registry in `mlir/bench/Generate.cpp`.
4. Add the explicit Python types in a family registration source under
   `bindings/bench/`, register its direct submodule in `register_bench.cpp`, and
   add the source to `bindings/bench/CMakeLists.txt`.
5. Test the reference, strict instance specification JSON, emitter structure,
   `jeff` conversion, and Python generation.

Do not add a second catalog, a generic option map, or a public base class.

## Reproducibility contract

Instance specifications reject unknown fields and invalid values. The case ID
does not depend on a path or output format. Parsing a manifest checks its
resolved parameters, logical output, reference, definition version, and case ID.
Before evaluation, normalize backend results to the manifest's big-endian
`result` order.
