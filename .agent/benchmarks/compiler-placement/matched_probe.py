"""Compare the complete device pipeline using identical saved QC inputs."""

import argparse
import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path
from statistics import median
from time import perf_counter

from mqt.core.bench import qpe, repeat_until_success
from mqt.core.mlir import QCProgram, compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--label", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
root = Path(__file__).parent
device = open_device("mqt.ddsim.default")
result = {
    "label": args.label,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "native_modules": {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in sys.modules.items()
        if name.startswith("mqt.") and str(getattr(module, "__file__", "")).endswith(".so")
    },
    "repeats": args.repeats,
    "rows": [],
}
for family, widths in [("iterative", [8, 32]), ("standard", [8, 16, 32, 64]), ("rus", [4, 16, 64])]:
    for width in widths:
        source = root / f"{family}-{width}-input.mlir"
        program = QCProgram.from_mlir_file(source)
        row = {"family": family, "width": width, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        durations = []
        failures = []
        for _ in range(args.repeats):
            start = perf_counter()
            try:
                compiled = compile_program(program, target=device)
            except (RuntimeError, ValueError) as error:
                failures.append(str(error))
            durations.append(perf_counter() - start)
        if failures:
            assert len(failures) == args.repeats, (family, width, failures)
            row |= {"error": failures[0], "failure_seconds": durations}
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            continue
        row |= {
            "compile_seconds": durations,
            "compile_median": median(durations),
            "bitcode_bytes": len(compiled.payload),
        }
        if width <= 16:
            benchmark = (
                repeat_until_success.RepeatUntilSuccess(repeat_until_success.Options(data_qubits=width))
                if family == "rus"
                else qpe.QPE(
                    qpe.Options(
                        precision=width,
                        phase=Fraction(3, 8),
                        method=qpe.Method.ITERATIVE if family == "iterative" else qpe.Method.STANDARD,
                    )
                )
            )
            durations = []
            for _ in range(args.repeats):
                start = perf_counter()
                job = device.submit_job(compiled.payload, ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=1024, custom1=17)
                job.wait()
                raw = job.get_counts()
                durations.append(perf_counter() - start)
                counts = raw if family == "rus" else {bits[::-1]: count for bits, count in raw.items()}
                tvd = benchmark.evaluate(counts).total_variation_distance
                assert tvd < (0.03 if family == "rus" else 1e-12), (family, width, counts, tvd)
            row |= {
                "execution_seconds": durations,
                "execution_median": median(durations),
                "shots": 1024,
                "seed": 17,
                "counts": counts,
                "tvd": tvd,
            }
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
