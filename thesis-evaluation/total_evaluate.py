import evaluate
import qiskit_run

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import json
import os
import numpy as np
import subprocess

OUT_DIR = "./figures"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

QISKIT_CACHE_FILE = "./evaluate_cache_qiskit.json"
QISKIT_RUST_CACHE_FILE = "./evaluate_cache_qiskit-rust.json"
MQT_CACHE_FILE = "./evaluate_cache_mqt.json"


def average_results(
    all_results: list[dict[str, dict[str, int | float]]],
) -> dict[str, dict[str, int | float]]:
    result = {}
    for r in all_results:
        for benchmark_name, measurements in r.items():
            if not benchmark_name in result:
                result[benchmark_name] = {}
            for metric_name, value in measurements.items():
                if not metric_name in result[benchmark_name]:
                    result[benchmark_name][metric_name] = 0.0
                result[benchmark_name][metric_name] += value / ITERATIONS
    return result


if os.path.exists(QISKIT_CACHE_FILE):
    with open(QISKIT_CACHE_FILE, "r") as f:
        qiskit_results = json.load(f)
else:
    ITERATIONS = 20
    all_results = []
    for _ in range(ITERATIONS):
        all_results.append(qiskit_run.evaluate())

    # [benchmark_name -> [metric_name -> value]]
    qiskit_results = average_results(all_results)

    with open(QISKIT_CACHE_FILE, "w") as f:
        json.dump(qiskit_results, f)

if os.path.exists(MQT_CACHE_FILE):
    with open(MQT_CACHE_FILE, "r") as f:
        mqt_results = json.load(f)
else:
    ITERATIONS = 20
    all_results = []
    for _ in range(ITERATIONS):
        # evaluate.evalue() will only process statistics files; need to re-run mqt-cc using run.sh
        subprocess.run(["./run.sh"]).check_returncode()
        all_results.append(evaluate.evaluate())

    # [benchmark_name -> [metric_name -> value]]
    mqt_results = average_results(all_results)

    with open(MQT_CACHE_FILE, "w") as f:
        json.dump(mqt_results, f)

if os.path.exists(QISKIT_RUST_CACHE_FILE):
    with open(QISKIT_RUST_CACHE_FILE, "r") as f:
        qiskit_rust_results = json.load(f)
else:
    ITERATIONS = 20
    all_results = []
    for _ in range(ITERATIONS):
        all_results.append(qiskit_run.evaluate(evaluate_rust_timings=True))

    # [benchmark_name -> [metric_name -> value]]
    qiskit_rust_results = average_results(all_results)

    with open(QISKIT_RUST_CACHE_FILE, "w") as f:
        json.dump(qiskit_rust_results, f)

print("In MQT, but not Qiskit: ", mqt_results.keys() - qiskit_results.keys())
print("In Qiskit, but not MQT: ", qiskit_results.keys() - mqt_results.keys())


x: dict[str, list[str]] = {}
y1: dict[str, list[int | float]] = {}
y2: dict[str, list[int | float]] = {}


def define_division_metric(
    new_metric: str, old_metric: str, divisor_metric: str, benchmark_name: str
):
    y1[new_metric] = y1.get(new_metric, []) + [
        m[old_metric] / m[divisor_metric] if m[divisor_metric] > 0 else float("nan")
    ]
    y2[new_metric] = y2.get(new_metric, []) + [
        q[old_metric] / q[divisor_metric] if q[divisor_metric] > 0 else float("nan")
    ]
    x[new_metric] = x.get(new_metric, []) + [benchmark_name]


aliases_qiskit = {
    "totalSingleQubitDecompositions": "successfulSingleQubitDecompositions",
    "totalTwoQubitDecompositions": "successfulTwoQubitDecompositions",
}
aliases_mqt = {
    "timeInCircuitCollection": "timeInCircuitCollectionStandalone",
}
names = sorted(mqt_results.keys() & qiskit_results.keys())
for name in names:
    m = mqt_results[name]
    # q = qiskit_results[name]
    q = qiskit_rust_results[name]

    for metric in m.keys() | q.keys():
        if metric in m:
            y1[metric] = y1.get(metric, []) + [m[metric]]
        if metric in aliases_mqt:
            y1[aliases_mqt[metric]] = y1.get(aliases_mqt[metric], []) + [m[metric]]
        if metric in q:
            y2[metric] = y2.get(metric, []) + [q[metric]]
        if metric in aliases_qiskit:
            y2[aliases_qiskit[metric]] = y2.get(aliases_qiskit[metric], []) + [
                q[metric]
            ]

    define_division_metric(
        "timePerSingleQubitDecomposition",
        "timeInSingleQubitDecomposition",
        "totalSingleQubitDecompositions",
        name,
    )
    define_division_metric(
        "timePerTwoQubitDecomposition",
        "timeInTwoQubitDecomposition",
        "totalTwoQubitDecompositions",
        name,
    )

    for metric in y1.keys() | y2.keys():
        x[metric] = names

titles = {
    "subCircuitComplexityChange": "Complexity Improvement after Decomposition",
    "successfulSingleQubitDecompositions": "Number of Successful Single-Qubit Decompositions",
    "successfulTwoQubitDecompositions": "Number of Successful Two-Qubit Decompositions",
    "timeInCircuitCollection": "Sub-Circuit Collection Time [µs]",
    "timeInCircuitCollectionStandalone": "Sub-Circuit Collection Time [µs]",
    "timeInSingleQubitDecomposition": "Total Time for Single-Qubit Decompositions [µs]",
    "timeInTwoQubitDecomposition": "Total Time for Two-Qubit Decompositions [µs]",
    "totalCircuitCollections": "Number of Sub-Circuit Collections",
    "totalSingleQubitDecompositions": "Total Number of Single-Qubit Decompositions",
    "totalTouchedGates": "Total Number of Gates in Collected Sub-Circuits",
    "totalTwoQubitDecompositions": "Total Number of Two-Qubit Decompositions",
    "timePerTwoQubitDecomposition": "Time / Two-Qubit Decomposition [µs]",
    "timePerSingleQubitDecomposition": "Time / Single-Qubit Decomposition [µs]",
    "twoQubitCreationTime": "Time for Creation of Two-Qubit Basis Decomposers [µs]",
}
legend_positions = {
    "totalTouchedGates": "upper left",
    "timeInCircuitCollection": "upper right",
    "timeInCircuitCollectionStandalone": "upper left",
    "timePerSingleQubitDecomposition": "upper left",
    "timePerTwoQubitDecomposition": "lower right",
    "totalTouchedGates": "upper left",
    "twoQubitCreationTime": "lower right",
}
pruneFunctions = {
    "timeInCircuitCollection": lambda value: np.isclose(value, 0),
    "timeInCircuitCollectionStandalone": lambda value: np.isclose(value, 0),
    "timeInSingleQubitDecomposition": lambda value: np.isclose(value, 0),
    "timeInTwoQubitDecomposition": lambda value: np.isclose(value, 0),
    "timePerTwoQubitDecomposition": lambda value: np.isclose(value, 0),
    "timePerSingleQubitDecomposition": lambda value: np.isclose(value, 0),
    "twoQubitCreationTime": lambda value: np.isclose(value, 0),
}
# modifications = {
#     "subCircuitComplexityChange": lambda value: -1.0 * value,
# }
for metric in x.keys():
    plt.title(titles.get(metric, metric))
    # x_values = x[metric] # use for benchmark names on x axis
    x_values = [str(i) for i in range(len(x[metric]))]

    # if metric in modifications:
    #     for y in y1[metric]:
    #         y = modifications[metric](y)
    #     for y in y2[metric]:
    #         y = modifications[metric](y)

    DEFAULT_POINT_SIZE = 100
    scale1 = []
    scale2 = []
    num_erased_y = 0
    for i in range(len(y1[metric])):
        if metric in y1:
            y1i = y1[metric][i - num_erased_y]
        else:
            y1i = None
        if metric in y2:
            y2i = y2[metric][i - num_erased_y]
        else:
            y2i = None
        if (
            (y1i == None and y2i == None)
            or (y1i != None and y2i != None and math.isnan(y1i) and math.isnan(y2i))
            or (
                y1i != None
                and y2i != None
                and pruneFunctions.get(metric, lambda _: False)(y1i)
                and pruneFunctions.get(metric, lambda _: False)(y2i)
            )
        ):
            x_values.pop(i - num_erased_y)
            if metric in y1:
                y1[metric].pop(i - num_erased_y)
            if metric in y2:
                y2[metric].pop(i - num_erased_y)
            num_erased_y += 1
        elif (
            y1i == None
            or math.isnan(y1i)
            or pruneFunctions.get(metric, lambda _: False)(y1i)
        ):
            scale1.append(0)
            scale2.append(DEFAULT_POINT_SIZE)
        elif (
            y2i == None
            or math.isnan(y2i)
            or pruneFunctions.get(metric, lambda _: False)(y2i)
        ):
            scale1.append(DEFAULT_POINT_SIZE)
            scale2.append(0)
        else:
            scale1.append(DEFAULT_POINT_SIZE)
            scale2.append(DEFAULT_POINT_SIZE)

    ymin = float("inf")
    if metric in y1:
        mqt_scatter = plt.scatter(
            x_values, y1[metric], color="blue", s=scale1, alpha=0.4
        )
        mqt_scatter.set_label("MQT")
        ymin = min(ymin, min(y1[metric]))
    if metric in y2:
        qiskit_scatter = plt.scatter(
            x_values, y2[metric], color="red", s=scale2, alpha=0.4
        )
        qiskit_scatter.set_label("Qiskit")
        ymin = min(ymin, min(y2[metric]))

    # plt.xticks(x_values) # does not work for strings
    plt.xticks(rotation=45)
    if ymin > 0:
        # let matplotlib handle non-positive values automatically
        plt.ylim(bottom=0)

    yint = []
    locs, labels = plt.yticks()
    for each in locs:
        yint.append(int(each))
    plt.yticks(yint)

    leg = plt.legend(loc=legend_positions.get(metric, "best"))
    for handle in leg.legend_handles:
        handle.set_sizes([DEFAULT_POINT_SIZE])
    plt.savefig(
        f"{OUT_DIR}/{metric}.pdf", format="pdf", bbox_inches="tight", pad_inches=0
    )
    #plt.show()
    plt.clf()

for i, name in enumerate(names):
    if "_indep_1_none_O0" in name:
        print(
            i, " & \\code{", name.removesuffix("_indep_1_none_O0"), "} & 1\\\\", sep=""
        )
    elif "_indep_2_none_O0" in name:
        print(
            i, " & \\code{", name.removesuffix("_indep_2_none_O0"), "} & 2\\\\", sep=""
        )
    else:
        print("UNKNOWN")

for metric in x.keys():
    print()
    print(
        f"Average MQT {metric}:",
        np.average(np.ma.masked_invalid(y1[metric])) if metric in y1 else "-",
    )
    print(
        f"Average Qiskit {metric}:",
        np.average(np.ma.masked_invalid(y2[metric])) if metric in y2 else "-",
    )
    print(
        f"Median MQT {metric}:",
        np.median(np.ma.masked_invalid(y1[metric])) if metric in y1 else "-",
    )
    print(
        f"Median Qiskit {metric}:",
        np.median(np.ma.masked_invalid(y2[metric])) if metric in y2 else "-",
    )
