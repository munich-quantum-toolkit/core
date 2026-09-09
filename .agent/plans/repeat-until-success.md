# Scalable repeat-until-success benchmark

Status: complete.

## Scope and decisions

Generalize Paetznick and Svore Figure 8 to the Pauli string
`P = X tensor ... tensor X` on `data_qubits` data qubits plus one ancilla. The
validated C++ options own the width bound, reused by JSON, Python, and
generation. Default width one preserves the original circuit semantics.
Canonical JSON resolves the width explicitly and includes it in case IDs. This
is unreleased definition version 1.

Both controlled-X gates become structured controlled-P sweeps. Since P is a
Hermitian involution, success implements `(I + i sqrt(2) P)/sqrt(3)` with
probability 3/4, and failure preserves arbitrary data up to global phase. The
failure path only resets the measured ancilla with X. The retry loop is
unbounded. A backend may impose its own execution budget.

The final Y/X parity readout retains the one-bit distribution
`P(0) = 1/2 + sqrt(2)/3`. Full bitstring output would make empirical TVD require
exponentially many shots. Width scales gate work and entanglement, but the
rank-two state remains easy for decision diagrams and expected T count is 8/3.
The one-million data-qubit cap follows the existing bounded-width benchmark
policy and is not a backend capacity guarantee.

## Validation

Check strict JSON and typed width bounds, resolved default and width-dependent
case IDs, Python options and generated stubs, the exact attempt/recovery
structure, seeded execution at widths 1, 2, 5, and 32, compact generation at the
maximum width, and a multi-qubit jeff round trip. Use 16,384 shots and a
case-specific TVD tolerance of 0.01: the binomial tail bound is below 7e-12,
while an all-zero sampler has TVD about 0.0286. Other benchmark tolerances
remain unchanged. Run native benchmark tests and the MLIR benchmark binary, then
the repository lint and C++ lint sessions.

Local results: all 57 native and 29 MLIR benchmark tests pass; all 54 Python
benchmark tests pass after rebuilding bindings and regenerating stubs. The CLI
regression test and 32-data-qubit generation in both QC and jeff formats pass.
General lint and full-file C++ lint pass for the committed change.

## Follow-ups

Keep three separate benchmark families for later work:

- Gearbox/PAR circuits: scale arity or nested retries; independently validate
  recovery and amplitude distortion for entangled controls.
- Magic-state distillation: start with 15-to-1 or Bravyi–Haah blocks; define
  noisy input states, acceptance, conditional infidelity, and resources per
  accepted state. Discarding failed candidates differs from preserving RUS data.
- Magic-state cultivation: scale code distance, checking rounds, and growth;
  define noise and syndrome semantics first. Validate phases independently of
  Clifford surrogates and account for the authors' circuit errata.

These families are outside this implementation. Research sources and derivations
are retained in the repeat-until-success audit.
