# Historical Slurm performance evidence

Status: historical; the current deployment is described in the
[Slurm integration plan](slurm-integration.md).

At baseline `552feffdbf5e2396660bb3e91a21936a9d7adf63`, local compiler caching
reduced a clean wheel build from 63.27 to 41.90 seconds with 62 hits from 67
requests. Image size fell from 516 to 450 MB. These measurements describe that
revision and environment, not the current cluster.

Native and Python adapter tests, runner tests, and real Slurm tests passed for
that change, including simultaneous independent runs. Hosted cache behavior was
not measured. The [historical audit](../audits/slurm-performance-2026-09-08.md)
contains the experiment details.
