from pathlib import Path
from mqt.core.mlir import QCOProgram, OutputFormat, compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device
root=Path(__file__).parent
program=QCOProgram.from_mlir_file(root/'partial-release.mlir')
compiled=compile_program(program,output=OutputFormat.QIR_ADAPTIVE)
(root/'partial-release.ll').write_text(compiled.ir)
job=open_device('mqt.ddsim.default').submit_job(compiled.to_bitcode(),ProgramFormat.QIR_ADAPTIVE_MODULE,num_shots=1,custom1=17)
job.wait()
try:
    print(job.get_counts())
except RuntimeError as error:
    print('Observed release failure:',error)
else:
    raise AssertionError('Release failure did not reproduce')
