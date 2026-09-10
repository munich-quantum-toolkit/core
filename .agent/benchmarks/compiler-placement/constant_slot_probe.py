from pathlib import Path
from mqt.core.mlir import QCOProgram, OutputFormat, compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device
p = QCOProgram.from_mlir_file(Path(__file__).with_name('constant-slot-loop.mlir'))
d = open_device('mqt.ddsim.default')
try:
    compile_program(p, target=d)
    print('Device compilation passed')
except ValueError as error:
    print('Device compilation:', error)
c = compile_program(p, output=OutputFormat.QIR_ADAPTIVE)
job = d.submit_job(c.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=1, custom1=17)
job.wait()
counts = job.get_counts()
assert counts == {'1': 1}, counts
print('Direct Adaptive QIR execution:', counts)
