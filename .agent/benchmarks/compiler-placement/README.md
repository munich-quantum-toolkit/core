# Indexed placement measurements

The matched comparison uses the complete device-directed compilation pipeline
and identical saved QC inputs. Both versions query the packaged DDSIM target,
perform placement and synthesis, and produce Adaptive QIR.

- Before: `7fca20764e0f4b4fbb995fd6e2f759c74275c6c1`.
- After: `a1d7ec8ccb7a6489b731197ef94eba90de3f44f6` (implementation before the final upstream rebase).
- CPython 3.14.7, GCC 13, LLVM/MLIR 23.1.0, MinSizeRel Python extensions,
  DGX Spark AArch64. Exact platform and native module SHA-256 hashes are in each
  raw JSON file. Later formatting, dialect registration, and lint fixes do not
  affect these measured paths.
- Three alternating before/after batches, three samples each: nine samples per
  workload and variant. This shared host also ran other builds. Ranges show the
  resulting timing variation; small differences are not evidence of speedups.
- Execution includes submission, JIT/job setup, waiting, and counts retrieval.
  Each sample uses 1,024 shots and seed 17. QPE returns the exact phase 3/8; RUS
  is checked against the analytic parity distribution with TVD below 0.03.
  Widths 32 and 64 are compile-only. This is not a runtime simulator benchmark.

Values are median [minimum, maximum] milliseconds; bytes are before / after.

| Input | Compile before | Compile after | Bitcode bytes | Execute before | Execute after |
| --- | ---: | ---: | ---: | ---: | ---: |
| iterative 8 | 47.64 [39.11, 102.14] | 55.77 [36.66, 86.33] | 3612 / 3612 | 37.89 [18.25, 44.26] | 39.10 [37.84, 44.65] |
| iterative 32 | 44.26 [37.42, 92.90] | 52.63 [37.31, 97.07] | 3632 / 3632 | - | - |
| standard 8 | 60.87 [42.29, 104.36] | 46.32 [38.53, 97.18] | 3888 / 3624 | 41.33 [22.01, 56.05] | 55.18 [25.71, 66.27] |
| standard 16 | 62.59 [57.17, 127.84] | 85.99 [42.79, 104.41] | 5424 / 3740 | 90.25 [65.23, 161.85] | 95.14 [69.03, 174.92] |
| standard 32 | 161.22 [120.29, 280.52] | 44.82 [43.40, 96.47] | 11332 / 3988 | - | - |
| standard 64 | 386.36 [368.17, 840.71] | 51.09 [43.69, 106.84] | 22876 / 4480 | - | - |
| rus 4 | 47.87 [39.70, 86.58] | 46.73 [39.94, 95.38] | 3640 / 3748 | 32.05 [31.34, 37.90] | 34.17 [15.76, 39.45] |
| rus 16 | 50.22 [47.64, 103.04] | 45.24 [40.75, 102.01] | 4184 / 3916 | 42.53 [40.29, 88.26] | 72.84 [40.63, 98.74] |
| rus 64 | 89.44 [82.78, 97.46] | 53.46 [42.94, 106.39] | 6612 / 4624 | - | - |

![Matched compilation, size, and execution measurements](matched-performance.png)

Standard QPE at width 64 has a clear compilation and size improvement. There is
no universal runtime improvement: loops add execution work, and RUS at width 16
has a higher execution median here. RUS at width 4 also grows slightly in bytecode.
The site table still grows linearly with register width. The long-loop native
regression separately verifies that 100,001 iterations retain one loop body.

## Reproduce

Build each measured revision in a separate environment using the same
MinSizeRel configuration. The compressed patches preserve the measured source
across rebases: apply `measured-before.patch.gz` to `ad74680f1` for the baseline,
then `measured-after.patch.gz` for the replacement (decompress with `gzip -dc`
and pass the result to `git apply`). Use an empty QDMI configuration file and set
`MQT_CORE_QDMI_CONFIG_JSON` to `{"schema-version":1,"qdmi":{"devices":[]}}`.
The packaged DDSIM device remains available. Run, alternating variants for each
of three batches:

```sh
/path/to/environment/bin/python .agent/benchmarks/compiler-placement/matched_probe.py \
  --label FULL_REVISION --output .agent/benchmarks/compiler-placement/matched-before-1.json --repeats 3
uv run --no-project --with matplotlib python .agent/benchmarks/compiler-placement/summarize.py
```

Use `matched-after-N.json` for the after variant and N=1,2,3 for the batches.
`matched_probe.py` verifies output quality on every executed sample;
`summarize.py` checks matching input hashes and stable module hashes within each
variant. Matplotlib is needed only for this one-off plot, not by MQT Core.

`baseline.json`, `execution.json`, `probe.py`, `execute_probe.py`, and the saved
`*-direct.ll` files belong to the earlier device-versus-direct diagnostic. They
bypass target passes in one arm and must not be used as optimization evidence.
`partial_release_probe.py` and its LLVM output reproduce the old ownership
failure; current native regression tests cover its repaired behavior.
