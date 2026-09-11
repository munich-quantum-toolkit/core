# MLIR audit resolution

Status: complete; locally validated after rebasing onto
`ba3aea8b1b618d8b000629c3995224ef4f73fa75`, including #2505.

Retain the ownership fixes, exact comparator checks, diagnostic lifetime,
reproducer support, and verifier-owned malformed-input coverage. Limit tensor
access permutations to distinct constant slots; retain QC disposal permutations
and remove obsolete QIR runtime-name exceptions. Keep normal custom passes with
preparation and cleanup in one pass manager, and consume QIR register
descriptors at finalization. Do not expand into general alias analysis or
semantic matching.

The [audit record](../audits/mlir-tests-diagnostics.md) contains the ranked
findings, surviving contracts, regression evidence, and validation.
