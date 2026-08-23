---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Structured quantum benchmarks

MQT Core defines a benchmark as validated, benchmark-specific parameters plus an
analytic reference. One instance produces a structured QC program, a resolved
manifest, and a stable case ID. Each benchmark returns one classical register
named `result`. Outcome strings are big-endian: the highest-index result bit is
the leftmost character.

## Discover the catalog

The command-line registry is the current list of available families. Each family
has its own JSON Schema.

```{code-cell} ipython3
import json
import subprocess


def run_bench(*arguments: str) -> dict[str, object]:
    """Run mqt-core-bench and parse its JSON output."""
    completed = subprocess.run(
        ["mqt-core-bench", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


catalog = run_bench("list")
[entry["id"] for entry in catalog["benchmarks"]]
```

```{code-cell} ipython3
qft_schema = run_bench("describe", "qft")
qft_schema["properties"]["parameters"]
```

## Configure a typed instance

Python exposes benchmark-specific option types. The QFT input below is the
uniform superposition of multiples of two. Both circuit methods use the same
logical output and reference.

```{code-cell} ipython3
from mqt.core.bench import QFT, QFTMethod, QFTOptions


benchmark = QFT(
    QFTOptions(
        qubits=3,
        period_exponent=1,
        method=QFTMethod.SEMICLASSICAL,
    )
)
benchmark.options.method, benchmark.output.name, benchmark.output.width
```

Python also accepts {py:class}`fractions.Fraction` for QPE phases. It reduces
the phase modulo one turn before it enters the C++ API. The reduced denominator
must fit in an unsigned 64-bit integer.

## Inspect the canonical request and manifest

Canonical JSON records every resolved default. A manifest also binds the logical
output, reference descriptor, family-definition version, and case ID.

```{code-cell} ipython3
request = json.loads(benchmark.request_json)
manifest = json.loads(benchmark.manifest_json)
request, {
    "case_id": manifest["case_id"],
    "outputs": manifest["outputs"],
    "reference": manifest["reference"],
}
```

## Query and evaluate the reference

For three output bits and period exponent one, QFT has two equal peaks.

```{code-cell} ipython3
{outcome: benchmark.probability(outcome) for outcome in ("000", "100", "010")}
```

```{code-cell} ipython3
evaluation = benchmark.evaluate({"000": 500, "100": 500})
{
    "total_variation_distance": evaluation.total_variation_distance,
    "squared_hellinger_fidelity": evaluation.squared_hellinger_fidelity,
    "success_probability": evaluation.success_probability,
}
```

Total variation distance zero and squared Hellinger fidelity one identify an
exact distribution. Bernstein--Vazirani and Grover also report the observed
success probability.

## Generate structured IR

Generation returns the same `QCProgram` type as the MLIR compiler bindings. The
program can enter the normal compiler pipeline.

```{code-cell} ipython3
program = benchmark.generate()
assert program.is_valid
structured_ir = program.ir
assert program.copy().to_qco().is_valid
structured_ir
```

## Run the command-line workflow

The CLI writes the program first and its manifest last. A manifest is therefore
the completion marker. Existing output files always cause an error.

```{code-cell} ipython3
import tempfile
from pathlib import Path


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    request_path = root / "request.json"
    counts_path = root / "counts.json"
    request_path.write_text(benchmark.request_json, encoding="utf-8")
    counts_path.write_text(
        json.dumps({"schema_version": 1, "counts": {"000": 5, "100": 5}}),
        encoding="utf-8",
    )

    generated = run_bench(
        "generate",
        "--request",
        str(request_path),
        "--format",
        "qc",
        "--output",
        str(root / "generated"),
    )
    result = run_bench(
        "evaluate",
        "--manifest",
        generated["manifest_path"],
        "--counts",
        str(counts_path),
    )

    cli_round_trip = {
        "program_suffix": Path(generated["program_path"]).suffixes,
        "case_ids_match": generated["case_id"] == result["case_id"],
        "shots": result["shots"],
        "metrics": result["metrics"],
    }

cli_round_trip
```

Use `--format jeff` instead of `--format qc` to write a binary `jeff` program.
The output format changes the file name, but not the semantic case ID.

## C++ API

The installed `MQT::CoreBench` target provides typed parameters, references,
evaluation, requests, and manifests.

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
`mqt::bench::generate(...)` overloads from `mlir/Bench/Generate.h` and returns a
`mlir::QCProgram`. This target is not installed until MQT Core installs the
wider MLIR compiler API.

## Add a benchmark

Adding a family requires five explicit extension points:

1. Add typed options, validation, an analytic reference, and evaluation under
   `include/mqt-core/bench/` and `src/bench/`.
2. Add its schema and evaluation callback to the private semantic registry in
   `src/bench/JSON.cpp`.
3. Add one structured emitter and one request callback to the private MLIR
   registry in `mlir/benchmark/Generate.cpp`.
4. Add the explicit Python types in `bindings/bench/register_bench.cpp`.
5. Test the reference, strict JSON, emitter structure, `jeff` conversion, and
   Python generation.

Do not add a second catalog, a generic option map, or a public base class.

## Reproducibility contract

Requests reject unknown fields and invalid values. The case ID does not depend
on a path or output format. Parsing a manifest checks its resolved parameters,
logical output, reference, definition version, and case ID. Before evaluation,
normalize backend results to the manifest's big-endian `result` order.
