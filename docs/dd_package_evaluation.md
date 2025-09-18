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

After running the target, you will see a `results_<your_argument>.json` file in your build directory that contains all the data collected during the benchmarking process. An exemplary `results_<your_argument>.json` file might look like this:

```{code-cell} ipython3
import json
from pathlib import Path

filepath = Path("../test/python/dd/evaluation/results_baseline.json")

with filepath.open(mode="r", encoding="utf-8") as f:
    data = json.load(f)
    json_formatted_str = json.dumps(data, indent=2)
    print(json_formatted_str)
```

To compare the performance of your newly proposed changes to the existing implementation, the benchmark script should be executed once based on the branch/commit you want to compare against and once in your new feature branch. Make sure to pass different arguments as different file names while running the target (e.g. `baseline` and `feature`).

+++

## Running the comparison

There are two ways to run the comparison. Either you can use the Python module {py:mod}`mqt.core.dd_evaluation` or you can use the CLI.
Both ways are shown below.
In both cases, you need to have the `evaluation` extra of `mqt-core` (i.e., `mqt.core[evaluation]`) installed.

+++

### Using the Python package

The Python package provides a function {py:func}`~mqt.core.dd_evaluation.compare` that can be used to compare two generate result files.
The function takes two arguments, the file path of the baseline json and the file path of the json results from your changes.
The function will then print a detailed comparison. An exemplary run is shown below.

```{code-cell} ipython3
from mqt.core.dd_evaluation import compare

baseline_path = "../test/python/dd/evaluation/results_baseline.json"
feature_path = "../test/python/dd/evaluation/results_feature.json"
compare(baseline_path, feature_path)
```

Note that the method offers several parameters to customize the comparison. See {py:func}`mqt.core.dd_evaluation.compare`.
An exemplary run adjusting the parameters is shown below.

```{code-cell} ipython3
compare(baseline_path, feature_path, no_split=True)
```

### Using the CLI

In an even simpler fashion, the comparison can be run from the command line via the automatically installed CLI.
Examples of such runs are shown below.

```{code-cell} ipython3
! mqt-core-dd-compare ../test/python/dd/evaluation/results_baseline.json ../test/python/dd/evaluation/results_feature.json --factor=0.2 --only_changed
```

```{code-cell} ipython3
! mqt-core-dd-compare ../test/python/dd/evaluation/results_baseline.json ../test/python/dd/evaluation/results_feature.json --no_split --dd --task=functionality
```

```{code-cell} ipython3
! mqt-core-dd-compare ../test/python/dd/evaluation/results_baseline.json ../test/python/dd/evaluation/results_feature.json --dd --algorithm=bv --num_qubits=1024
```
