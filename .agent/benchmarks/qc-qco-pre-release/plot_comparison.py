"""Plot matched production comparisons, including measured regressions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = Path(__file__).parent
rows = json.loads((here / 'comparison.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
for ax, workload, title in zip(axes.flat, ['used', 'fresh', 'extracted', 'empty'],
        ['Required resets on used slots', 'Removable resets on fresh slots',
         'Preparing registers with extracted qubits', 'Preparing registers without extractions']):
    sizes = sorted({r['size'] for r in rows if r['workload'] == workload})
    for variant, label in [('baseline', 'Upstream ad74680f1'), ('fixed', 'Patched')]:
        quantiles = np.array([np.quantile([r['ms'] for r in rows
            if r['workload'] == workload and r['size'] == size and r['variant'] == variant],
            [.25, .5, .75]) for size in sizes])
        ax.plot(sizes, quantiles[:, 1], '-o', label=label)
        ax.fill_between(sizes, quantiles[:, 0], quantiles[:, 2], alpha=.2)
    ax.set(xscale='log', yscale='log', xlabel='Slots / registers', ylabel='Time (ms)', title=title)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('QC/QCO fixes · matched medians and interquartile ranges · CPU 18')
fig.savefig(here / 'comparison.png', dpi=170)
