"""Compare the saved upstream binary with the patched binary."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess
import sys

here = Path(__file__).parent
builder_only = '--builder-only' in sys.argv
rows = [r for r in json.loads((here / 'comparison.json').read_text()) if r['workload'] in ['fresh', 'used']] if builder_only else []
for workload in ([] if builder_only else ['fresh', 'used']):
    for size in [128, 256, 512, 1024]:
        expected = None
        for pair in range(5):
            versions = ['baseline', 'fixed'] if pair % 2 == 0 else ['fixed', 'baseline']
            for version in versions:
                output = here / f'{workload}-{size}-{version}.out.mlir'
                result = subprocess.run(
                    ['taskset', '-c', '18', str(here / f'probe-{version}'),
                     'canonicalize-qco', 'qasm', str(here / 'inputs' / f'{workload}-{size}.mlir'),
                     '3', str(output)], capture_output=True, text=True, check=True, timeout=60,
                )
                source = output.read_text()
                digest = hashlib.sha256(source.encode()).hexdigest()
                if expected is None:
                    expected = digest
                assert digest == expected, (workload, size, version, 'output differs')
                assert source.count('qco.reset ') == (size if workload == 'used' else 0)
                for sample in json.loads(result.stdout)['samples']:
                    rows.append(dict(workload=workload, size=size, variant=version,
                                     pair=pair, ms=sample['ms'][0], ops=sample['ops'], sha256=digest))
        print(workload, size, {v: round(statistics.median(r['ms'] for r in rows
              if r['workload'] == workload and r['size'] == size and r['variant'] == v), 3)
              for v in ['baseline', 'fixed']}, flush=True)
for kind in ['empty', 'extracted']:
    for size in [64, 256, 1024, 4096]:
        for pair in range(5):
            versions = ['baseline', 'fixed'] if pair % 2 == 0 else ['fixed', 'baseline']
            for version in versions:
                result = subprocess.run(
                    ['taskset', '-c', '18', str(here / f'probe-{version}'),
                     'builder-prep', kind, str(size), '3'],
                    capture_output=True, text=True, check=True, timeout=60,
                )
                for sample in json.loads(result.stdout)['samples']:
                    rows.append(dict(workload=kind, size=size, variant=version,
                                     pair=pair, ms=sample['ms'][0], ops=sample['ops']))
        print(kind, size, {v: round(statistics.median(r['ms'] for r in rows
              if r['workload'] == kind and r['size'] == size and r['variant'] == v), 3)
              for v in ['baseline', 'fixed']}, flush=True)
(here / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')

outputs = {}
for kind in ['scalar', 'tensor']:
    samples = []
    for _ in range(24):
        result = subprocess.run([str(here / 'probe-fixed'), 'builder', kind, '16', '1'],
                                capture_output=True, text=True, check=True, timeout=60)
        samples.append(dict(sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                            source=result.stdout))
    assert len({s['sha256'] for s in samples}) == 1, kind
    outputs[kind] = samples
    print(kind, '1 distinct output from 24 processes', flush=True)
(here / 'determinism-fixed.json').write_text(json.dumps(outputs, indent=2) + '\n')
