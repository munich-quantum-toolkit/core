from pathlib import Path
import csv
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'timings.csv').open() as stream:
    rows = list(csv.DictReader(stream))
fig, ax = plt.subplots(figsize=(6.5, 4))
for mode, label in [('baseline', 'Before'), ('structural', 'Strict structural'), ('permutation', 'Permutation helper fast path')]:
    sizes = [1000, 2000, 4000]
    samples = [[float(row['milliseconds']) for row in rows
                if int(row['gates']) == size and row['comparator'] == mode]
               for size in sizes]
    medians = [statistics.median(values) for values in samples]
    ax.errorbar(sizes, medians,
                yerr=[[median - min(values) for median, values in zip(medians, samples)],
                      [max(values) - median for median, values in zip(medians, samples)]],
                marker='o', capsize=3, label=label)
ax.set(xlabel='Gates in a verified QCO chain', ylabel='Comparison time (ms, logarithmic scale)',
       yscale='log', title='Identical structure: seven samples per point')
ax.set_xticks([1000, 2000, 4000])
ax.grid(True, which='major', alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(root / 'comparison.png', dpi=170)
