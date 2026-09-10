"""Plot recorded medians without rerunning measurements."""
import csv
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).parent
before = json.loads((root / 'before.json').read_text())['results']
after_path = root / 'after.json'
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for path, label in [(root / 'before.json', 'Before'), (after_path, 'After')]:
    if not path.exists():
        continue
    data = json.loads(path.read_text())['results']
    names = ['snapshot', 'compiled_submit', 'source_submit']
    axes[0].plot(names, [data[x]['median'] * 1000 for x in names], 'o-', label=label)
    names = [f'chain_construct_{x}' for x in (256, 1024, 4096)]
    axes[1].loglog([256, 1024, 4096], [data[x]['median'] * 1000 for x in names], 'o-', label=label)
rows = list(csv.DictReader((root / 'tuples.csv').open()))
for method in ['permutation', 'sorted_views']:
    axes[2].loglog([1000, 4000, 8000], [statistics.median(float(r['seconds']) for r in rows if int(r['size']) == n and r['method'] == method) * 1000 for n in (1000, 4000, 8000)], 'o-', label=method)
for ax, title in zip(axes, ['DDSIM Bell workflow', 'Chain target construction', 'Reordered tuple comparison']):
    ax.set_title(title)
    ax.set_ylabel('Median elapsed time (ms)')
    ax.legend()
    ax.grid(alpha=.25)
axes[0].tick_params(axis='x', rotation=15)
axes[1].set_xlabel('Sites')
axes[2].set_xlabel('Tuples')
fig.savefig(root / 'comparison.png', dpi=160)
