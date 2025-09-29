# GPT-5 Agent Prompt: Integrate Discussions Summary (v3) into `quantum-dialect-revamp-plan.md`

## Mission

Synchronize the existing dialect plan (`quantum-dialect-revamp-plan.md`) with the authoritative design decisions encoded in `mlir-discussions-summary.json` (version 3), while _respecting and reusing_ all still-relevant material from the existing plan. Produce a fully revised, internally consistent markdown document that becomes the new single source of truth.

You MUST:

- Use BOTH the existing plan AND the JSON spec as inputs.
- Treat the JSON (version 3) as authoritative where conflicts arise.
- Preserve valuable explanatory prose from the current plan unless it contradicts decisions.
- Update, relocate, or condense sections for clarity—but do not silently drop useful context.
- Ensure EVERY MLIR code example is syntactically valid, semantically coherent, and consistent with the described semantics (no stale modifier order, no missing yields in value semantics, no invalid types).
- Fail the task (i.e., do not produce a final doc) if you cannot ensure MLIR validity; instead, explicitly list corrections needed.

---

## Authoritative Decisions (From JSON v3 — Must Be Reflected)

1. Canonical modifier nesting order (outer → inner): `ctrl`, `negctrl`, `pow`, `inv`.
2. Negative controls (`negctrl`) are first-class; DO NOT auto-normalize them into positive controls in canonical passes.
3. Named single-qubit gates are retained; implement _bidirectional_ canonicalization:
   - Named Simplification: `U3/U2/U1 → simplest named gate` (tolerance-based).
   - Universal Expansion: `named gate → canonical U3` (backend-optional).
4. Matrix-based gates ARE part of the MVP (no deferral); practical usage up to 3 qubits; optimize 2×2 and 4×4 cases.
5. Matrix unitary attribute: flat, row-major dense array of complex numbers.
6. Normalization (modifier ordering, merging, identity elimination) ALWAYS runs before other transformations (integrated into canonicalization).
7. Testing strategy: googletest builder-based structural/semantic equivalence is PRIMARY; textual FileCheck / lit tests limited to parser/printer smoke cases.
8. Sequences are control-neutral (always report 0 controls).
9. Interface currently does NOT expose explicit enumeration for pow/inv layers (deferred need).
10. No negative-control rewriting pass in MVP.
11. Open questions list is EMPTY for MVP—new ambiguities must be clearly introduced if discovered.

---

## Required Output

Produce TWO artifacts in a single response:

1. The **fully revised** `quantum-dialect-revamp-plan.md` (complete file content; no partial diff).
2. A trailing JSON change summary object:
   ```json
   {
     "decisionSyncVersion": 3,
     "addedSections": [...],
     "modifiedSections": [...],
     "removedSections": [...],
     "notes": [...],
     "mlirExampleAudit": {
       "checkedExamples": N,
       "invalidExamplesFound": 0
     }
   }
   ```

If any MLIR example cannot be validated conceptually (e.g., invalid region signature, missing yield, wrong result count), set `"invalidExamplesFound" > 0` and DO NOT produce the revised document—produce instead a diagnostic section listing every problematic snippet and how to fix it.

---

## Transformation Tasks (Execute in Order)

1. **Parse Inputs**
   - Load and internalize the JSON spec (v3).
   - Read the existing markdown plan fully.
   - Build an internal map of sections: Overview, Architecture, Types, Base Gates, Modifiers, Sequences, Interface, User-Defined Gates, Builders, Canonicalization/Passes, Testing, Conversions, Conclusion.

2. **Conflict Audit**
   Identify and correct in the existing plan:
   - Outdated canonical order (`ctrl → pow → inv` must become `ctrl → negctrl → pow → inv`).
   - Any implication that matrix-based gates are "future" or "optional".
   - Any claim or example suggesting negative controls are normalized away.
   - Missing mention of bidirectional canonicalization.
   - Over-reliance on FileCheck in testing rationale (must be re-scoped).
   - Redundant or conflicting descriptions of modifiers.

3. **Structural Enhancements**
   Add / modify sections:
   - "Design Decisions (Synchronized with Discussions Digest v3)" near the top (concise bullet list referencing decisions above).
   - "Pass Inventory" with a table:
     | Pass | Purpose | Phase | Idempotent | Mandatory | Notes |
     Include: NormalizationPass, Named Simplification Pass, Universal Expansion Pass, Matrix Unitarity Validation Pass.
     Add a "Deferred" subsection listing: Matrix decomposition, Basis gate registry, Extended trait inference.
   - Integrate matrix-unitary support into user-defined gates AND core architecture (not just an isolated advanced feature).

4. **Modifiers Section Rewrite**
   - Ensure examples reflect canonical nesting order.
   - Provide at least one nested example:
     ```mlir
     mqtref.ctrl %c0, %c1 {
       mqtref.negctrl %c2 {
         mqtref.pow {exponent = 2.0 : f64} {
           mqtref.inv {
             mqtref.x %t
           }
         }
       }
     }
     ```
   - Provide corresponding value-semantics form if appropriate, ensuring `yield` operands/results match region signatures.

5. **Matrix Gates Integration**
   - Document attribute form with a valid example:
     ```mlir
     // 2x2 example (Pauli-Y)
     %u = mqtref.matrix %q0 {
       matrix = dense<[[0.0+0.0i, 0.0-1.0i],
                       [0.0+1.0i, 0.0+0.0i]]> : tensor<2x2xcomplex<f64>>
     }
     ```
     Adjust syntax to match your dialect conventions (if a different op naming scheme like `mqtref.unitary` is used, align accordingly).
   - Show 4×4 example (two-qubit) with shape validation.
   - Clarify unitarity validation pass (tolerance-based; optional strict mode).

6. **Interface Section Adjustments**
   - Remove or annotate any obsolete pow/inv enumeration ambitions.
   - Clarify control counting (wrappers aggregate; seq = 0; matrix + composite definitions = 0 controls themselves unless wrapped).

7. **Canonicalization Section Update**
   - Split into:
     - "Normalization (Early Canonical Form)"
     - "Named Simplification"
     - "Universal Expansion"
     - "Matrix Validation"
     - (Deferred) "Matrix Decomposition"
   - Remove obsolete examples (e.g., automatic negctrl erasure).
   - Ensure algebraic simplification examples only include decisions currently endorsed (e.g., `inv(inv(X))`, `pow(RZ(pi/2), 2)`).

8. **Testing Strategy Rewrite**
   Must include:
   - Structural equivalence assertion workflow: build → run canonical pipeline → compare IR structurally (ignore SSA names).
   - Idempotence test pattern (run Normalization twice).
   - Round-trip (parser → printer → parser) smoke tests for: base gates, modifiers, matrix gate, nested modifiers.

9. **MLIR Example Validation**
   For every example:
   - Ensure region form is valid: block arguments present if operands thread through (value semantics).
   - Ensure all yields match enclosing op result arity/types.
   - Ensure attributes syntactically valid (e.g., `exponent = 0.5 : f64`).
   - Ensure tensor element types use `complex<f64>` where required.
   - Ensure no stray `%q0_new` without use or yield in SSA form examples.

10. **Conclusion Update**
    - Reflect finalized scope: MVP includes matrix ops, dual canonicalization, early normalization, builder-first strategy.
    - Remove speculative phrasing about deferring matrix support.

11. **Change Summary JSON**
    - Populate arrays precisely (section titles or approximate headings).
    - Notes should list any semantic clarifications (e.g., replacing previous canonical order, introducing Pass Inventory).

---

## Style & Consistency Rules

- Prefer concise declarative sentences in decision summary sections.
- Avoid re-defining the same rationale in multiple places—link or reference earlier sections.
- Use consistent dialect namespace prefixes (`mqtref.` / `mqtopt.`). If any example mixes them incorrectly, fix it.
- Use fenced code blocks with `mlir` language tag for MLIR IR.
- Do not include TODOs for MVP features—only for explicitly deferred future items under a "Deferred" heading.

---

## Explicit Prohibitions

DO NOT:

- Invent new passes or features not sanctioned by JSON v3.
- Remove valuable context silently—if removed for redundancy, capture that in the change summary's `removedSections` or `notes`.
- Produce partial file content.
- Output explanations after the final JSON summary.

---

## Validation Checklist (Self-Check Before Emitting Output)

Mark each internally (do not include the checklist in final output):

- All examples MLIR-valid.
- No contradictions about negative controls.
- Matrix gates clearly part of MVP.
- Pass inventory present & accurate.
- Testing section prioritizes builder-based structural tests.
- Canonical modifier order consistent across narrative + examples.
- No obsolete canonicalization rules remain.
- JSON summary structurally valid.

---

## Final Output Format

1. Full revised markdown (beginning at first line with title—retain or improve existing heading).
2. JSON change summary object (standalone; no extra prose after it).

If ANY blocking inconsistency remains → output ONLY a diagnostic section titled `BLOCKED` with a bullet list of unresolved issues (no partial plan).

BEGIN.
