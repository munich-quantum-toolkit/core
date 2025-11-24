---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Benchmarking the DD Package

MQT Core provides a benchmark suite to evaluate the performance of the DD package.
This can be especially helpful if you are working on the DD package and want to know how your changes affect its performance.

+++

## Generating results

In order to generate benchmark data, MQT Core provides the `mqt-core-dd-eval` CMake target (which is made available by passing `-DBUILD_MQT_CORE_BENCHMARKS=ON` to CMake). This target will run the benchmark suite and generate a JSON file containing the results. To this end, the target takes a single argument which is used as a component in the resulting JSON filename.

+++

After running the target, you will see a `results_<your_argument>.json` file in your build directory that contains all the data collected during the benchmarking process. An exemplary `results_<your_argument>.json` file might look like this (taken from the `eval` directory):

```{code-cell} ipython3
import json
from pathlib import Path

filepath = Path("../eval/results_baseline.json")

with filepath.open(mode="r", encoding="utf-8") as f:
    data = json.load(f)
    json_formatted_str = json.dumps(data, indent=2)
    print(json_formatted_str)
```

To compare the performance of your newly proposed changes to the existing implementation, the benchmark script should be executed once based on the branch/commit you want to compare against and once in your new feature branch. Make sure to pass different arguments as different file names while running the target (e.g. `baseline` and `feature`).

+++

## Running the comparison

The MQT Core source code contains a script that can be used to compare the results of two runs of the benchmark suite.
The script is located in `eval/dd_evaluation.py`.
It uses PEP 723 inline script metadata to specify the script's dependencies.

The comparison can be run from the command line via:

```{code-cell} ipython3
! ../eval/dd_evaluation.py ../eval/results_baseline.json ../eval/results_feature.json --factor=0.2 --only_changed
```

```{code-cell} ipython3
! ../eval/dd_evaluation.py ../eval/results_baseline.json ../eval/results_feature.json --no_split --dd --task=functionality
```

```{code-cell} ipython3
! ../eval/dd_evaluation.py ../eval/results_baseline.json ../eval/results_feature.json --dd --algorithm=bv --num_qubits=1024
```

This internally runs `uv run --script --quiet` on the respective benchmark script (including automatically installing any missing dependencies).
