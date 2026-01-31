import matplotlib.pyplot as plt
import glob
import os

INPUT_DIR = "./tmp-output"

def read_stats(file_name) -> dict[str, int]:
    result = {}

    with open(file_name, "r") as f:
        for line in f.readlines():
            line = line.lstrip()
            line.removeprefix("(S)")
            line = line.lstrip()
            elements = line.split(" ")
            elements = list(filter(lambda x: x != " " and len(x) > 0, elements))

            metric = elements[2]
            value = int(elements[1])
            result[metric] = value if value < 2**31 else value - 2**32

    return result

def evaluate():
    all_stats = {}
    for file in glob.glob(f"{INPUT_DIR}/*.statistic"):
        print(f"Processing {file}...")
        name = os.path.basename(file).removesuffix(".statistic")
        name = name.removesuffix(".qasm")
        name = name.removesuffix(".mlir")
        all_stats[name] = read_stats(file)

    print(all_stats)
    return all_stats
