from pathlib import Path
import json, shlex, subprocess, sys
root=Path.cwd()
build=root/'build/release-clang-ipo'
audit=root/'.agent/benchmarks/qc-qco-pre-release'
entries=json.loads((build/'compile_commands.json').read_text())
entry=next(e for e in entries if '/lib/Conversion/QCOToQC/' in e['file'])
def compile(source,output,which='QCOToQC'):
 if which in ['QCToQCO','QCOToQC']:
  e=next(e for e in entries if f'/lib/Conversion/{which}/' in e['file'])
  args=shlex.split(e['command'])
 else:
  e=next(e for e in entries if e['file'].endswith(which) or ('/Unity/' in e['file'] and f'/{which}"' in Path(e['file']).read_text()))
  args=shlex.split(e['command'])
  if '/Unity/' in e['file']:
   import re
   unity=Path(e['file']).read_text()
   unity=re.sub(r'(?<=#include ")[^"]*/'+which+r'(?=")',str(source),unity)
   source=audit/f'{mode}-unity-{which}';source.write_text(unity)
 if mode == 'upstream-index': args.insert(1, '-I'+str(audit/'upstream-include'))
 args[args.index('-o')+1]=str(output)
 args[args.index('-c')+1]=str(source)
 subprocess.run(args,cwd=build,check=True)
commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','mqt-cc'],text=True)
line=commands.splitlines()[-1]
args=shlex.split(line)
if args[:2]==[':', '&&']: args=args[2:]
if '&&' in args: args=args[:args.index('&&')]
args=[x for x in args if not x.endswith('.o')]
mode=sys.argv[1] if len(sys.argv)>1 else 'baseline'
if mode == 'noreset':
 source=(root/'mlir/lib/Dialect/QTensor/IR/Operations/ExtractOp.cpp').read_text()
 old='FoldExtractAfterInsertPattern, RemoveResetAfterExtract'
 assert old in source
 (audit/'noreset-ExtractOp.cpp').write_text(source.replace(old,'FoldExtractAfterInsertPattern'))

if mode == 'upstream-index':
 header=Path('mlir/include/mqt/Dialect/QCO/Builder/QCOProgramBuilder.h')
 saved=audit/'upstream-include'/header.relative_to('mlir/include')
 saved.parent.mkdir(parents=True,exist_ok=True)
 saved.write_bytes(subprocess.check_output(['git','show',f'ad74680f1:{header}']))
 source='mlir/lib/Dialect/QCO/Builder/QCOProgramBuilder.cpp'
 (audit/'upstream-index-QCOProgramBuilder.cpp').write_bytes(subprocess.check_output(['git','show',f'ad74680f1:{source}']))
probe_source=audit/(sys.argv[2] if len(sys.argv)>2 else 'probe.cpp')
probe_obj=audit/f'{mode}-{probe_source.stem}.o'
compile(probe_source,probe_obj)
args.insert(1,str(probe_obj))
args.insert(2,str(build/"mlir/lib/Dialect/QCO/Builder/libMLIRQCOProgramBuilder.a"))
for which in ['QCToQCO','QCOToQC','Pipeline.cpp','ShrinkRegisters.cpp','ShrinkQubitRegisters.cpp','ExtractOp.cpp','QCOProgramBuilder.cpp']:
 p=audit/f'{mode}-{which if which.endswith('.cpp') else which+'.cpp'}'
 if p.exists():
  out=p.with_suffix('.o');compile(p,out,which);args.insert(1,str(out))
args[args.index('-o')+1]=str(audit/f'probe-{mode}')
subprocess.run(args,cwd=build,check=True)
print(audit/f'probe-{mode}')
