---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Structured quantum benchmarks

MQT Core defines a benchmark as validated, benchmark-specific parameters plus an
analytic reference. The same instance generates a structured QC program, a
resolved manifest, and a stable case ID. All current benchmarks return one
classical register named `result`. Outcome strings are big-endian: the
highest-index result bit is the leftmost character.

## Benchmark families

| ID       | Parameters                                                                                                                                                                      | Reference                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `ghz`    | `qubits` from 1 through 1,000,000; `topology` is `linear` or `star`; `basis` is `z` or `x`. The defaults are `linear` and `z`. X-basis references support at most 1,075 qubits. | Z-basis outcomes are all zero or all one. X-basis outcomes have even parity. The topology changes preparation, not the reference. |
| `grover` | A big-endian `marked_bitstring` of width 2 through 62; `iterations` is an optional integer from 0 through 2,147,483,647. MQT Core selects the best iteration count by default.  | The single marked outcome and the uniform residual distribution. Evaluation also reports the observed marked-outcome probability. |
| `qpe`    | `precision` from 1 through 1,000,000; an exact rational `phase` in turns; `method` is `standard` or `iterative`. The default method is `standard`.                              | The phase-estimation distribution. Both methods use the same reference and result order.                                          |

Run `mqt-core-bench describe <id>` to obtain the authoritative JSON Schema for a
request.

## Python

Python accepts {py:class}`fractions.Fraction` phases of any size and reduces
them modulo one turn. The reduced denominator must fit in an unsigned 64-bit
integer. Canonical requests contain that reduced value.

```{code-cell} ipython3
import json
from fractions import Fraction

from mqt.core.benchmarks import QPE, QPEMethod, QPEOptions

benchmark = QPE(
    QPEOptions(
        precision=3,
        phase=Fraction(1, 8),
        method=QPEMethod.ITERATIVE,
    )
)

assert json.loads(benchmark.request_json)["parameters"] == {
    "method": "iterative",
    "phase": {"denominator": 8, "numerator": 1},
    "precision": 3,
}
assert benchmark.output.name == "result"
assert benchmark.output.width == 3
assert benchmark.probability("001") == 1.0

evaluation = benchmark.evaluate({"001": 1000})
assert evaluation.total_variation_distance == 0.0
assert evaluation.squared_hellinger_fidelity == 1.0
assert evaluation.success_probability is None

program = benchmark.generate()
assert program.is_valid
```

`GHZOptions`, `GroverOptions`, and `QPEOptions` expose typed keyword arguments.
Each benchmark provides `probability`, `evaluate`, `request_json`,
`manifest_json`, `case_id`, and `generate`.

## Command line

List the fixed registry or inspect one request schema:

```console
mqt-core-bench list
mqt-core-bench describe ghz
```

Create `request.json`:

```json
{"schema_version":1,"benchmark":"ghz","parameters":{"qubits":3,"topology":"linear","basis":"z"}}
```

Generate textual QC-dialect MLIR in `generated/`:

```console
mqt-core-bench generate --request request.json --format qc --output generated
```

Use `--format jeff` to generate a binary `jeff` program instead. The command
writes the program and a format-specific manifest. It refuses to replace either
file unless `--overwrite` is present.

To evaluate results, create `counts.json` with logical, big-endian outcomes:

```json
{"schema_version":1,"counts":{"000":500,"111":500}}
```

Then evaluate the counts against the generated instance:

```console
mqt-core-bench evaluate --manifest generated/ghz-*.qc.manifest.json --counts counts.json
```

The result reports 1,000 shots, total variation distance zero, squared Hellinger
fidelity one, and no success probability. Only Grover defines a success outcome.

## C++ reference API

The installed `MQT::CoreBenchmarks` target owns typed parameters, references,
requests, and manifests. It does not expose the MLIR emitter. Use Python or
`mqt-core-bench` to generate structured programs.

```cpp
#include "benchmarks/Grover.hpp"

#include <cassert>

int main() {
  const mqt::benchmarks::Grover benchmark{{.markedBitstring = "101"}};
  const auto evaluation = benchmark.evaluate({{"101", 1000}});
  assert(evaluation.successProbability == 1.0);
}
```

Link the target from CMake:

```cmake
find_package(mqt-core CONFIG REQUIRED)
target_link_libraries(my-benchmark PRIVATE MQT::CoreBenchmarks)
```

## Reproducibility contract

Requests reject unknown fields and invalid values. Canonical requests and
manifests record all resolved defaults. A manifest binds the parameters, logical
output, analytic reference, definition version, and case ID. Parsing a manifest
checks all of those fields again.

The case ID does not depend on a path or output format. QC and `jeff` programs
for the same instance therefore use the same case ID. Before evaluation,
normalize backend results to the manifest's big-endian `result` order. An exact
distribution has total variation distance zero and squared Hellinger fidelity
one.
