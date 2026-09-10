from pathlib import Path
p=Path(__file__).parent/'inputs';p.mkdir(exist_ok=True)
for n in [128,256,512,1024,2048]:
 for kind in ['fresh','used']:
  lines=['module {','func.func @main() attributes {mqt.entry_point} {',f'%size = arith.constant {n} : index',f'%t0 = qtensor.alloc(%size) : tensor<{n}x!qco.qubit>']
  for i in range(n):lines += [f'%i{i} = arith.constant {i} : index']
  chain=0
  for phase in (['prepare','reset'] if kind=='used' else ['reset']):
   for i in range(n):
    prefix=f'{phase}{i}'
    lines += [f'%e{prefix}, %q{prefix} = qtensor.extract %t{chain}[%i{i}] : tensor<{n}x!qco.qubit>']
    q=f'%q{prefix}'
    if phase=='reset':lines += [f'%r{prefix} = qco.reset {q} : !qco.qubit -> !qco.qubit'];q=f'%r{prefix}'
    lines += [f'%h{prefix} = qco.h {q} : !qco.qubit -> !qco.qubit']
    chain+=1;lines += [f'%t{chain} = qtensor.insert %h{prefix} into %e{prefix}[%i{i}] : tensor<{n}x!qco.qubit>']
  lines += [f'qtensor.dealloc %t{chain} : tensor<{n}x!qco.qubit>','return','}','}']
  (p/f'{kind}-{n}.mlir').write_text('\n'.join(lines)+'\n')
