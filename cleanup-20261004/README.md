🤖 *AI text below* 🤖

# Reproducing the cleanup evaluation

The report identifies the two compiler versions and the source-isolation build.
`source-verification.json` records their extension hashes. Worker metadata
records toolchain flags and environment; its checkout HEAD describes the live
workspace, while the explicit source/binary mapping identifies the preserved
baseline. Package version strings can predate commit creation.

Install each revision in its own environment using the normal Core development
build. Preserve the previous environment before rebuilding the current one.
Set the checkout/interpreter paths in `large.py`, `sweep_cleanup.py`, and
`profile_cleanup.py` to these local environments. Keep the frozen `corpus/`
unchanged; its manifest includes input hashes and exact target capabilities.

Run `python sweep_cleanup.py` for 398 native-contract/quality checks and the
separate ten-case, five-sample paired timings. Run `python profile_cleanup.py`
for the five before/after pass profiles. Do not run builds, tests, or other
benchmarks concurrently. The sweep rejects overlap with compiler processes.
The profile runner was also run on an idle host; its stdout/stderr is retained.

Run `uv run --with matplotlib --no-project python analyze_cleanup.py`, then
`python write_cleanup_report.py`, to regenerate the tables and plots. The
report's historical count comparison uses the preceding evaluation's raw rows,
bundled under `historical/`; update the `OLD` path in the analysis script to that
directory when running the standalone bundle.

`small-validation/` records the 400 semantic/export checks using the earlier
small corpus, bundled in `small-corpus/`. `evaluate-small-validation.py` is the
exact harness version used for those checks. The current `evaluate.py` adds
RZ-excluded depth for the new paired experiment.

The `diagnostics/` directory contains temporary rewrite-listener counts,
canonicalizer-setting probes, and rebuilt-source gate-count probes. Instrumented
pattern timings and exploratory settings were used only to locate work, not
as end-to-end performance measurements. No compiled binaries are distributed.

All artifacts are separate from the source PR. SHA-256 hashes in
`artifact-manifest.json` cover the bundle contents.
