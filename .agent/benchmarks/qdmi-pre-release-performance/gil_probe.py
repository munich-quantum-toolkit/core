import statistics
import threading
import time
from itertools import pairwise

from mqt.core.mlir import CompilerTarget, OutputFormat, compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

target = CompilerTarget(
    2,
    connectivity=CompilerTarget.Connectivity.all_to_all(),
    native_operations=CompilerTarget.NativeOperations.unrestricted(),
)
device = open_device("mqt.ddsim.default")


def measure(label, fn):
    durations = []
    gaps = []
    for _ in range(3):
        stop = threading.Event()
        ready = threading.Event()
        times = []

        def heartbeat(stop=stop, ready=ready, times=times):
            while not stop.is_set():
                times.append(time.perf_counter())
                ready.set()
                time.sleep(0.001)

        thread = threading.Thread(target=heartbeat)
        thread.start()
        ready.wait()
        time.sleep(0.01)
        start = time.perf_counter()
        _result = fn()
        durations.append(time.perf_counter() - start)
        time.sleep(0.01)
        stop.set()
        thread.join()
        gaps.append(max(b - a for a, b in pairwise(times)))
    print(
        label,
        "median_s",
        round(statistics.median(durations), 6),
        "heartbeat_gap_s",
        round(statistics.median(gaps), 6),
        "durations_s",
        durations,
        "gaps_s",
        gaps,
        flush=True,
    )


for n in [1000, 10000]:
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q;\n' + (
        "rx(0.1) q[0]; cx q[0],q[1]; rz(0.2) q[1];\n" * (n // 3)
    )
    typed = compile_program(source, output=OutputFormat.QCO)
    measure(
        f"explicit_source_{n}",
        lambda source=source: compile_program(
            source, target=target, program_format=ProgramFormat.QASM3
        ),
    )
    measure(
        f"explicit_typed_{n}",
        lambda typed=typed: compile_program(
            typed, target=target, program_format=ProgramFormat.QASM3
        ),
    )
    measure(
        f"device_source_{n}",
        lambda source=source: compile_program(
            source, target=device, program_format=ProgramFormat.QASM3
        ),
    )
    measure(
        f"device_typed_{n}",
        lambda typed=typed: compile_program(
            typed, target=device, program_format=ProgramFormat.QASM3
        ),
    )
