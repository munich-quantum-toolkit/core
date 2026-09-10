import json
import shlex
import subprocess
from pathlib import Path

evidence = Path(__file__).resolve().parent
root = evidence.parents[2]
build = root / "build/release"
entry = next(
    x
    for x in json.loads((build / "compile_commands.json").read_text())
    if x["file"].endswith("test_compiler_qdmi_adapter.cpp")
)
args = shlex.split(entry["command"])
args[args.index("-o") + 1] = str(evidence / "probe.o")
args[-1] = str(evidence / "probe.cpp")
subprocess.run(args, cwd=build, check=True)
cmds = subprocess.check_output(
    ["ninja", "-t", "commands", "mqt-core-mlir-unittests-compiler"],
    cwd=build,
    text=True,
).splitlines()
link = next(
    x
    for x in reversed(cmds)
    if " -o mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler " in x
)
args = shlex.split(link.split(" && ")[1])
args = [x for x in args if not x.endswith(".cpp.o")]
args.insert(1, str(evidence / "probe.o"))
args[args.index("-o") + 1] = str(evidence / "probe")
subprocess.run(args, cwd=build, check=True)
