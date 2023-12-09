"""Evaluating the json file generated by the benchmarking script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from os import PathLike

# Avoid output truncation
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

sort_options = ["ratio", "algorithm"]
higher_better_metrics = ["hits", "hit_ratio"]


def __flatten_dict(d: dict[Any, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary. Every value only has one key which is the path to the value."""
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(__flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def __post_processing(key: str) -> dict[str, str]:
    """Postprocess the key of a flattened dictionary to get the metrics for the DataFrame columns."""
    metrics_divided = key.split(".")
    result_metrics = {}
    if len(metrics_divided) < 4:
        raise ValueError("Benchmark " + key + " is missing algorithm, task, number of qubits or metric!")
    result_metrics["algorithm"] = metrics_divided.pop(0)
    result_metrics["task"] = metrics_divided.pop(0)
    result_metrics["num_qubits"] = metrics_divided.pop(0)
    num_remaining_benchmarks = len(metrics_divided)
    if num_remaining_benchmarks == 1:
        result_metrics["component"] = ""
        result_metrics["metric"] = metrics_divided.pop(0)
    elif num_remaining_benchmarks == 2:
        result_metrics["component"] = metrics_divided.pop(0)
        result_metrics["metric"] = metrics_divided.pop(0)
    else:
        separator = "_"
        # if the second-to-last element is not "total" then only the last element is the metric and the rest component
        if metrics_divided[-2] == "total":
            metric = separator.join(metrics_divided[-2:])
            result_metrics["metric"] = metric
            component = separator.join(metrics_divided[:-2])
            result_metrics["component"] = component
        else:
            result_metrics["metric"] = metrics_divided[-1]
            component = separator.join(metrics_divided[:-1])
            result_metrics["component"] = component

    return result_metrics


def __aggregate(baseline_filepath: str | PathLike[str], feature_filepath: str | PathLike[str]) -> pd.DataFrame:
    """Aggregate the data from the baseline and feature json files into one DataFrame for visualization."""
    base_path = Path(baseline_filepath)
    with base_path.open(mode="r", encoding="utf-8") as f:
        d = json.load(f)
    flattened_data = __flatten_dict(d)
    feature_path = Path(feature_filepath)
    with feature_path.open(mode="r", encoding="utf-8") as f:
        d_feature = json.load(f)
    flattened_feature = __flatten_dict(d_feature)

    for k, v in flattened_data.items():
        if k in flattened_feature:
            ls = [v, flattened_feature[k]]
            flattened_data[k] = ls
            del flattened_feature[k]
        else:
            ls = [v, "skipped"]
            flattened_data[k] = ls
    # If a benchmark is in the feature file but not in the baseline file, it should be added with baseline marked as
    # "skipped"
    for k, v in flattened_feature.items():
        ls = ["skipped", v]
        flattened_data[k] = ls

    before_ls, after_ls, ratio_ls, algorithm_ls, task_ls, num_qubits_ls, component_ls, metric_ls = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for k, v in flattened_data.items():
        after = v[1]
        before = v[0]
        if before in {"unused", "skipped"} or after in {"unused", "skipped"}:
            ratio = float("nan")
        else:
            ratio = after / before if before != 0 else 1 if after == 0 else 1000000000.0
        before_ls.append(before)
        after_ls.append(after)
        ratio_ls.append(ratio)

        # postprocessing
        result_metrics = __post_processing(k)
        algorithm_ls.append(result_metrics["algorithm"])
        task_ls.append(result_metrics["task"])
        num_qubits_ls.append(result_metrics["num_qubits"])
        component_ls.append(result_metrics["component"])
        metric_ls.append(result_metrics["metric"])

    df_all = pd.DataFrame()
    df_all["before"] = before_ls
    df_all["after"] = after_ls
    df_all["ratio"] = ratio_ls

    df_all["algorithm"] = algorithm_ls
    df_all["task"] = task_ls
    df_all["num_qubits"] = num_qubits_ls
    df_all["component"] = component_ls
    df_all["metric"] = metric_ls
    df_all.index = pd.Index([""] * len(df_all.index))

    return df_all


def compare(
    baseline_filepath: str | PathLike[str],
    feature_filepath: str | PathLike[str],
    factor: float = 0.1,
    only_changed: bool = True,
    sort: str = "ratio",
    no_split: bool = False,
) -> None:
    """Compare the results of two benchmarking runs from the generated json file.

    Args:
        baseline_filepath: Path to the baseline json file.
        feature_filepath: Path to the feature json file.
        factor: How much a result has to change to be considered significant.
        only_changed: Whether to only show results that changed significantly.
        sort: Sort the table by this column. Valid options are "ratio" and "experiment".
        no_split: Whether to merge all results together in one table or to separate the results into benchmarks that improved, stayed the same, or worsened.

    Returns:
        None
    Raises:
        ValueError: If factor is negative or sort is invalid.
        FileNotFoundError: If the baseline_filepath argument or the feature_filepath argument does not point to a valid file.
        JSONDecodeError: If the baseline_filepath argument or the feature_filepath argument points to a file that is not a valid JSON file.
    """
    if factor < 0:
        msg = "Factor must be positive!"
        raise ValueError(msg)
    if sort not in sort_options:
        msg = "Invalid sort option! Valid options are 'ratio' and 'algorithm'."
        raise ValueError(msg)

    df_all = __aggregate(baseline_filepath, feature_filepath)

    m1 = df_all["ratio"] < 1 - factor  # after significantly smaller than before
    m2 = df_all["metric"].str.endswith(tuple(higher_better_metrics))  # if the metric is "better" when it's higher
    m3 = df_all["ratio"] > 1 + factor  # after significantly larger than before
    m4 = (df_all["ratio"] != df_all["ratio"]) | ((1 - factor < df_all["ratio"]) & (df_all["ratio"] < 1 + factor))
    # ratio is NaN or after not significantly different from before

    if no_split:
        if only_changed:
            df_all = df_all[m1 | m3]
            print("All changed benchmarks:")
        else:
            print("All benchmarks:")

        df_all = df_all.sort_values(by=sort)
        print(df_all)
        return

    print("Benchmarks that have improved:")
    df_improved = df_all[(m1 & ~m2) | (m3 & m2)]
    df_improved = df_improved.sort_values(by=sort)
    print(df_improved)

    if not only_changed:
        print("Benchmarks that have stayed the same:")
        df_same = df_all[m4]
        df_same = df_same.sort_values(by=sort)
        print(df_same)

    print("Benchmarks that have worsened:")
    df_worsened = df_all[(m3 & ~m2) | (m1 & m2)]
    df_worsened = df_worsened.sort_values(by=sort)
    print(df_worsened)


def main() -> None:
    """Main function for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Compare the results of two benchmarking runs from the generated json files."
    )
    parser.add_argument("baseline_filepath", type=str, help="Path to the baseline json file.")
    parser.add_argument("feature_filepath", type=str, help="Path to the feature json file.")
    parser.add_argument(
        "--factor", type=float, default=0.1, help="How much a result has to change to be considered significant."
    )
    parser.add_argument(
        "--only_changed", action="store_true", help="Whether to only show results that changed significantly."
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="ratio",
        help="Sort the table by this column. Valid options are 'ratio' and 'algorithm'.",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Whether to merge all results together in one table or to separate the results into "
        "benchmarks that improved, stayed the same, or worsened.",
    )
    args = parser.parse_args()
    assert args is not None
    compare(args.baseline_filepath, args.feature_filepath, args.factor, args.only_changed, args.sort, args.no_split)
