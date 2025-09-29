# MLIR Quantum Dialect Discussions Digest

Purpose: Curated, compact, duplication‑free summary of prior threaded discussions to serve as a stable knowledge base for design, implementation, and future LLM queries.

---

## 1. Modifier & Control Semantics

### 1.1 Multiple Controls in a Single Modifier

- We allow a single `ctrl` (and `negctrl`) modifier to list multiple controls for compact IR and reduced traversal.
- Nesting remains valid but a normalization pass will flatten/merge where possible.

### 1.2 Canonical Modifier Ordering

Canonical order (outer → inner):

1. `ctrl`
2. `negctrl`
3. `pow`
4. `inv`

Rationale:

- Groups positive and negative controls first (most structural impact on qubit usage / control sets).
- Power and inverse are algebraic refinements on the underlying operation.
- Deterministic order enables straightforward structural hashing, CSE, pattern rewrites, and simplifies equality queries.

### 1.3 Negative Controls Representation

Current decision: keep distinct `ctrl` and `negctrl` wrappers (cleaner semantics than a combined mixed-control list variant).

Decision Update: Negative controls are first-class citizens; we WILL NOT automatically normalize them into positive controls in canonical pipelines. Transformations involving X gate sandwiches remain optional backend-specific rewrites, not part of normalization.

### 1.4 Control Counting Semantics (UnitaryInterface)

- Base (leaf) operations: 0 controls.
- `ctrl` / `negctrl`: number of (pos / neg) controls they introduce + recursively added controls of wrapped op.
- `pow`, `inv`: do not change control counts; they forward queries to the wrapped op.
- `seq` (sequence construct): always reports 0 controls (acts like a one-off composite/inline body rather than a modifier).
- User-defined unitary (definition): reports 0 controls.

Rationale: Control counting terminates cleanly; `seq` neutrality simplifies reasoning and prevents accidental double counting when sequences are wrapped by modifiers.

Open Question (closed): Interface exposure for enumerating `pow` / `inv` layers not needed now.

---

## 2. Gate Set vs. Single Universal Gate (U3-Only Approach)

Proposal Considered: Collapse to only a universal single-qubit gate (U3) + modifiers (OpenQASM 3 style).

Decision: Retain a rich set of named primitive gates (X, Y, Z, H, S, T, RX, RY, RZ, P, etc.) AND provide bidirectional canonicalization passes bridging named gates and universal U3 forms.

Rationale:

- Readability & ergonomic authoring.
- Numerical stability: Avoids always instantiating via floating-point U3 params (reduces drift / canonicalization complexity).
- Trait inference (e.g., Hermitian, Diagonal) is straightforward on symbolic gates; harder on generic parametrized forms.
- Alignment with existing ecosystems (OpenQASM library, Qiskit standard gates) while still permitting decomposition.
- Bidirectional passes satisfy both optimization introspection (simplify to named) and backend uniformity (expand to U3) requirements.

Committed Passes:

1. Named Simplification Pass: Simplify generic U3/U2/U1 into simplest named gates (tolerance-based) — deterministic, part of canonicalization pipeline.
2. Universal Expansion Pass: Expand named single‑qubit gates into canonical U3 (optionally gated by a flag / backend pipeline requirement).

Status: Dual canonicalization direction DECIDED (no longer an open question).

---

## 3. Matrix-Based Gate Definitions (Arbitrary Unitaries)

Decision: INCLUDED in initial specification (not deferred). No programmatic hard limit on qubit count; practical performance naturally constrains usage. Provide specialized fast paths and optimization focus for 1‑qubit (2×2) and 2‑qubit (4×4) matrices as part of the MVP (not postponed).

Representation:

- Attribute: Flat dense array (row-major) of complex numbers (candidate: MLIR DenseElementsAttr with complex element type). Potential specialized attributes may still be explored later if profiling indicates need.
- Optional metadata: dimension (inferred) and qubit arity (validation cross-check).

Validation:

- Unitarity Validation Pass: (Configurable) numerical tolerance check; may be skipped in trusted build modes.

Pros:

- Direct expression of synthesized results & externally imported unitaries.
- Unambiguous semantics and suitable anchor for decomposition passes.

Considerations:

- Large matrices quickly infeasible (exponential growth); accepted as user responsibility—no current heuristic advisory needed (≤3 qubits covers practical scope).
- Potential future compression (factorizations, tensor products) is out-of-scope for MVP.

---

## 4. Normalization & Canonicalization Passes (Current & Planned)

NormalizationPass (part of canonicalization pipeline; always scheduled before other transformations):

- Enforce modifier ordering: `ctrl` → `negctrl` → `pow` → `inv`.
- Merge adjacent identical/compatible modifiers (e.g., consecutive `ctrl` merges control sets; nested `pow` multiplies exponents; nested `inv` cancels).
- Eliminate neutral identities: `pow(X, 1) → X`, `inv(inv(X)) → X`.
- Simplify algebraic forms when safe (e.g., `pow(RZ(pi/2), 2) → RZ(pi)`).
- Flatten nested control wrappers.

Additional Passes:

- Gate inlining pass for user-defined unitary definitions until reaching standard basis gates.
- Named Simplification Pass (Section 2).
- Universal Expansion Pass (Section 2).
- Matrix Unitarity Validation Pass (Section 3).
- Future: Decomposition of matrix-based unitaries into chosen basis (configurable backend pass).

(Note: Negative control elimination explicitly omitted per Section 1.3 decision.)

---

## 5. Testing Strategy

Updated Direction (Stronger Commitment):

- Primary mechanism: googletest-based infrastructure constructing IR programmatically via builder API (see spec Section 9.2) and asserting structural / semantic equivalence.
- Textual `CHECK` tests reduced to minimal smoke tests for parser/round-trip coverage only.
- Emphasis on deterministic canonicalization pre-check (run normalization + canonicalization pipeline before equivalence assertions).

Guidelines:

- Provide helper utilities for: module creation, gate application, modifier wrapping, matrix unitary creation, control aggregation.
- Structural Equivalence: Use MLIR IR equivalence if available; otherwise implement recursive op/attribute/operand/result comparison normalized by canonical pass.
- Avoid reliance on incidental attribute ordering or SSA naming.

Action Items (refined):

1. Ensure comprehensive builder coverage (all ops, modifiers, matrix unitaries, unitary definitions).
2. Implement `assertEquivalent(afterPasses, expected)` helper.
3. Provide fixture functions for common gate patterns & modifier compositions.
4. Add round-trip (print→parse) smoke tests for representative IR forms (including matrix unitaries & modifiers nesting).

---

## 6. Default / Recommended Pass Pipelines

Baseline (tests & examples):

- `-canonicalize` (includes NormalizationPass via registration hook)
- `-remove-dead-values`
- Named Simplification Pass (enabled by default unless backend requests universal form)

Backend Universalization Pipeline (example):

- Baseline passes
- Universal Expansion Pass (if backend requires uniform U3)

Decision: Normalization (modifier ordering & merging) ALWAYS runs before any other transformation stages; integrated as early canonicalization pattern population.

---

## 7. Open Questions (Unresolved Summary)

Currently none. All previously tracked questions have been resolved for the MVP scope. New questions will re-open this section when they arise.

---

## 8. Decision Log (Chronological Core Decisions)

1. Keep distinct `ctrl` and `negctrl` modifiers (no combined mixed-control wrapper for now).
2. Establish canonical modifier ordering: `ctrl`, `negctrl`, `pow`, `inv`.
3. Represent sequences (`seq`) as control-neutral (0 controls) composites.
4. Retain named gate set; do not reduce to U3-only model.
5. Plan normalization & simplification passes mirroring legacy MQT Core canonicalization behavior.
6. Adopt builder-based structural testing strategy; textual tests become supplementary.
7. Recommend baseline pipeline: `-canonicalize` + `-remove-dead-values` (expand later).
8. Decide on bidirectional single-qubit canonicalization (Named Simplification & Universal Expansion passes).
9. Include matrix-based gate definitions in initial spec (flat dense array attribute; no hard size cap; unitarity validation pass).
10. Enforce normalization (modifier ordering + merging) always-before transformations via canonicalization integration.
11. Elevate googletest builder-based testing as primary; textual tests minimized.

---

## 9. Future Work (Backlog Candidates)

- Implement NormalizationPass (ordering, merging, identity elimination).
- Implement Named Simplification Pass & Universal Expansion Pass.
- Structural equivalence utilities & test harness helpers.
- Matrix-based unitary op implementation + unitarity validation pass.
- Performance specialization for 2×2 / 4×4 matrix attributes (fast paths).
- Gate definition inlining + matrix-to-basis decomposition passes.
- Trait inference for composites & matrix-based ops (Hermitian, Diagonal detection heuristics / numerical tolerance).
- Basis gate registry with canonical decomposition recipes.

(Removed: negative-control rewriting pass, apply_gate exploration.)

---

## 10. References to Legacy Logic

Legacy MQT Core IR already performs simplification of U3/U2/U1 to simpler gates (see former `StandardOperation::checkUgate` logic). New dialect passes replicate this within MLIR canonicalization infrastructure (Named Simplification Pass).

---

## 11. (Optional) Raw Discussion Log

The original threaded conversation has been condensed into this digest. Retrieve the prior version from version control if verbatim context is ever required.

---

End of digest.
