🤖 *AI text below* 🤖

# Matrix DD root-range evaluation

Review artifacts for the isolated matrix-range fix. Product changes are on
`codex/dd-matrix-root-range`; this separate branch contains only evaluation
artifacts.

- Baseline: `e7b37b7a42291a9816b735ef8ae2e53b0f37b990` (upstream main).
- Candidate: `15c106305557faeefa9c9f6d7f56e7ebf0c77269`.
- Environment: Linux ARM64 on DGX Spark, Clang 23.1.2, Release, CPU 16.
- Measurements: six ordinary matrix cases, three adjacent alternating-order
  trials per variant, plus six wide-Hadamard accuracy cases per variant.
- Ordinary cases: all 36 numerical checks passed. The candidate passed all
  six wide cases; upstream failed the four cases from 96 through 512 qubits.

Read the [report](matrix-evaluation-report.md) and
[method](matrix-evaluation-method.md). The JSONL files contain all 48 raw results;
the plans, summary, and verification files identify cases and expected
outcomes. The plots include observed trial ranges, not confidence intervals.

This is a numerical correctness fix. The measured performance varies by
workload: paired median CPU ratios range from 0.717 to 1.357, and paired peak
RSS ratios range from 0.998 to 1.008. These short, machine-specific trials do
not establish a general speedup. Binary64 range and the existing tolerance
for small local coefficients still apply.

Codex assisted with the implementation, evaluation, and these artifacts at
the maintainer's request.
