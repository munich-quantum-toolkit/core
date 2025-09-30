Design an LLM prompt for revising the RFC dialect proposal based on the following feedback:

## 1.1 Current State

- the existing dialects only implement the control (pos+neg) modifier, but no power or inverse modifier. That's the main drawback of the existing solution with regard to modifiers, not that they modifiers are encoded inconsistently
- The interfaces aren't really fragmented, they are just not expressive enough yet and do not cover all the use cases we would want to cover
- There is also no way currently to define custom gates, to get the unitary of a gate, or to compose multiple gates

## 1.2 Proposed direction

Aims to address all the drawbacks from the previous section

## 2.1 Dialect structure

Measurements and reset are not resource operations. They belong in their own category.
Resource operations should be the allocation (`alloc`) and deallocation (`dealloc`) of qubits as well as the definition of static qubits (`qubit`).

UnitaryExpr and UnitaryMatrix are actually the same concept. The proposal should be entirely phrased in terms of UnitaryExpr.

## 2.2 Principles

This section should be fleshed out a little more based on the remaining proposal

## 3.1 Quantum Types

all types should start with a capital letter (e.g., mqtref.Qubit) to not conflict with the static qubit allocation

## 3.2 Register Handling

should be fleshed out a little bit more based on the existing (working) implementation

## 7. Unified Interface Design

This section should come as section 4 before describing all various unitary operations and to clearly set the stage for what to expect from every following section on the unitary operations. Each following section should indicate how it expects to implement the various interface functions.

Combine the existing 7.1 and 7.2 into one single more elaborative proposal that includes more interface functions.
Especially functions for

- getting the i-th input operand, getting the i-th output operand, getting the i-th parameter (somehow handling the fact that the parameter might be defined as an attribute or a dynamic value). getting the output value corresponding to an input value and vice versa, get all parameters (combining static and dynamic parameters in the right order

## 4. Base Gate Operations

the `cx` gate is not a base gate, but only syntactic sugar that is explained later. Do not use it as an example of a two-qubit gate in this section, but rather use the parametrized rzz(theta) gate.

## 4.1 Philosophy

This should be fleshed out more. The base gates should heavily use the traits for target and parameter arity to define clean operations with efficient builders and custom assembly formats. These should be defined with as little code duplication as possible.
Each base gate defines an explicit UnitaryExpr representing its matrix. Gates without parameters always have a statically defined matrix. Those gates with parameters have a statically defined matrix representation if and only if all parameters are statically specified. Otherwise, the matrix representation depends on dynamic `mlir::Value`s

Parameters and qubit parameters should be distinguished in the text-based format similar to the existing implementation, where parameters are depicted in `(...)` directly after the gate name, with qubit arguments being listed as "regular" SSA values afterwards, e.g., `%q0_out = mqtopt.rx(%theta) %q0_in`. This should work with attributes and dynamic values.

## 5. Modifier Operations

Swap `negctrl` and `ctrl` in the canonical nesting order.

All subsections should have custom assembly format that enables more compact and MLIR idiomatic text-based IR descriptions. All subsections should also include examples in the reference and the value semantics dialects. These examples should be consistent and correct.

All modifiers should define canonicalization patterns.

## 5.1 Controls (`ctrl`) & Negative Controls (`negctrl`)

The unitary representation of these is the accordingly extended unitary of the underlying unitary operation.
It reports controls of the modifier plus any controls of the contained operation.
Parameters are just passed through.
Operands of the target operation are just passed through, control qubits are explicitly handled.
directly nested control modifiers are merged through canonicalization or folds
Control applied to a `seq` equals to a control applied to every gate of the sequence.
There may be an optional transformation pass that checks consecutive gates for shared controls and may extract these into a shared control modifier on a `seq`.

## 5.2 Power (`pow`)

Similar to the base gate parameters, `pow` modifiers should list the parameter in round brackets `(...)`.

The unitary of the pow modifier may be explicitly computed. Integer powers are just repetitions. Floating point values are computed based on the principal logarithm. Negative powers are canonicalized to an inverse modifier and a power modifier with a positive exponent. consecutive power modifiers are merged by multiplying the factors.
Specializations of powers of base gates with known analytical forms are to be implemented (e.g., rotation gates)

## 5.3 Inverse (`inv`)

The unitary of base gates may be explicitly computed (conjugate adjoint). This may be specialized for when the inverse is another explicit base gate as part of canonicalization. All qubit values are simply passed through.
The inverse of a sequence is the reversed sequence with every gate inverted. Consecutive `inv` cancel as part of the canonicalization pipeline

## 5.4 Nested Modifier Example (Canonical Order)

The example needs to be updated. It is also not yet correct as the blocks do not have the right block arguments (the nested sequence applies two controls to a single-qubit gate, so the outside modifier needs to have three qubit arguments.

## 5.5

Remove this subsection

## 6. Sequence Operations (`seq`)

Similar to Section 5, these operations should also have a custom assembly format that is as compact as possible and as idiomatic for MLIR as possible.

## 8. Matrix & User-Defined Gates

Should be written as if this was naturally part of the proposal and not being discussed on.
The matrix defined gates should also be proper definitions of gate symbols similar to section 8.2, i.e., they should contain a symbol name and it should be possible to apply them (similar to 8.2).
A unified interface for this would be appropriate so that the application operation can be reused between unitary and gate based definitions.
All of the operations should have convenient assembly formats.
These gates report no controls, expose their parameters, and their unitary is either directly available (unitary definition) or can be computed as a product of the unitaries in the definition (U_M,...U_1).

## 8.1 Matrix Unitary Operation

Tensors should be 2d shaped to reflect that these are matrices. these operations should also have a custom assembly format that is more compact.
Matrix definitions may also be parametrized by dynamic values.

## 9.1 Parser Sugar Examples

Provide more details on how these canonical expansions are actually realized in the MLIR framework.

## 9.2 9.2 C++ Builder (Sketch)

This builder is incredibly important for the test related changes. Ensure this is as easy to use as possible. Enrich the available interface by further methods based on the previous sections. This should not just be a sketch.

## 10 Canonicalization & Transformation Stages

Remove this section for now. Any canonicalization patterns should be directly described in the respective sections. They should not be dedicated passes, but integrated into the dedicated canonicalization pipeline in MLIR.
Any optional transformation passes should also be mentioned in the individual sections.

## 11. Pass Inventory

Similar to section 10, remove this section

## 12. Dialect Conversions

Remove this dedicated section but add the respective information about converting from one representation to the other to the individual sections on the operations. The individual sections should be almost self contained and contain detailed information on all relevant aspects.

## 13. Testing Strategy

Rephrase and iterate on this section based on feedback above

## 14. Analysis & Optimization Infrastructure

include the UnitaryMatrix subsection in a revised form based on feedback into a previous section that is suitable.

## 15. Canonical Identities & Algebraic Rules (Non-Exhaustive)

These should be part of the canonicalization subsections in previous sections and can be removed here after folding them into the previous sections.

## 16. Conclusion

Revamp the conclusions based on the updated proposal

---

Generated prompt:
You are an expert MLIR and quantum compiler technical writer. Rewrite the RFC for the quantum dialect revamp incorporating the structured feedback below. Produce a self‑contained, updated proposal (single document) with precise MLIR operation specifications, canonicalization rules inline, and consistent terminology.

Overall goals:
\- Clarify current limitations (only control modifiers exist; missing pow/inv; insufficient interface expressiveness; no custom gate definitions, composition, or unitary extraction).
\- Introduce a coherent UnitaryExpr abstraction used everywhere (replace any split between UnitaryMatrix and UnitaryExpr).
\- Reorder and restructure sections as specified.
\- Integrate canonicalization rules directly inside each operation/section (no separate pass inventory section).
\- Provide clear interface contracts before defining concrete ops.
\- Use capitalized types (e.g., mqtref.Qubit, mqtopt.Qubit).
\- Distinguish parameter operands (in parentheses) from qubit operands in examples: `%q_out = mqtopt.rx(%theta) %q_in`.
\- Provide both reference semantics (mqtref) and value semantics (mqtopt) examples for every unitary + modifier + sequence construct.
\- Ensure examples are correct: block arguments, result threading, modifier nesting order, etc.

Required new document structure (renumber all sections accordingly):

1. Overview and Goals
2. Current State and Limitations
3. Dialect Structure and Categories
   3.1 Resource Operations (alloc, dealloc, static qubit definition)
   3.2 Measurement and Reset (separate category)
   3.3 UnitaryExpr Concept (single abstraction)
4. Unified Unitary Interface Design
   \- Merge old 7.1 and 7.2; expand.
   \- Specify: identification, arities, controls (pos/neg), target access, parameter model (static + dynamic merged ordering), i\-th input/output access, mapping input↔output for value semantics, collecting all parameters, static vs dynamic unitary availability, matrix extraction, inversion, power, control extension hooks.
   \- Define how operations report: getNumTargets, getNumPosControls, getNumNegControls, getNumParams, getParameter(i), getInput(i), getOutput(i), mapOutputToInput(i), hasStaticUnitary, getOrBuildUnitaryExpr, etc.
5. Base Gate Operations
   5.1 Philosophy (traits for target/parameter arity, minimal duplication, explicit UnitaryExpr definition; static matrix when all params static; dynamic composition otherwise).
   5.2 Gate List (single\-qubit: x, y, z, h, s, sdg, t, tdg, rx, ry, rz, u, etc.)
   5.3 Multi\-qubit illustrative example: rzz(theta) instead of cx (cx deferred to sugar section).
   5.4 Syntax and Assembly Formats (parameter parentheses + qubits).
   5.5 Builders (static param overloads, dynamic param variants, combined convenience helpers).
   5.6 Canonicalization & Identities (parameter folding, specialization to simpler named gates).
   5.7 Value vs Reference Semantics mapping.
6. Modifier Operations
   6.1 Canonical Nesting Order (negctrl → ctrl → pow → inv) [swap per feedback].
   6.2 negctrl (semantics, unitary extension, merging/flattening).
   6.3 ctrl (same pattern; merging directly nested; seq lifting rule).
   6.4 pow (parentheses exponent, integer repetition, float principal log, negative exponent → inv+pow normalization, merge consecutive powers, analytical specializations).
   6.5 inv (adjoint computation, folding to known inverse base gates, double inverse elimination, sequence inversion reversal).
   6.6 Nested Example (correct threading, updated order, both dialects).
7. Sequence Operation (seq)
   \- Custom assembly format.
   \- Control lifting semantics (external control equivalence).
   \- Inversion + power behavior (defer to contained unitaries).
8. User Defined Gates & Matrix / Composite Definitions
   8.1 Symbolic matrix gate definitions (2D tensor shape, optional dynamic parameters).
   8.2 Composite gate definitions (sequence body; derive unitary via ordered product).
   8.3 Unified apply operation (applies any gate symbol: matrix or composite).
   8.4 Parameter handling (static/dynamic mix; ordering rules).
   8.5 Canonicalization (inline trivial identity, fold consecutive applies of same static gate if allowed).
   8.6 Unitary extraction rules (static matrix availability or lazy composition).
9. Parser & Builder Sugar
   9.1 Sugar expansions (cx, cz, ccx) → explicit modifier wrapping; specify how parse hooks lower to canonical form.
   9.2 Detailed builder API (fluent style): gate, param, controls (pos/neg), power, inverse, sequence, defineGate, defineMatrixGate, applyGate, withControls, withNegControls, withPow, withInv. Provide ergonomic overloads and RAII region helpers.
10. Testing Strategy
    \- Emphasize builder\-driven structural tests.
    \- Matrix correctness tests (unitarity, dimension).
    \- Interface conformance tests per op category.
    \- Canonicalization idempotence (run canonicalization twice).
    \- Sugar round trip tests.
    \- Negative tests (bad arity, mismatched params, invalid matrix shape, non\-unitary).
11. Integrated Canonicalization Rules Summary (collected references only; full definitions live inline above).
12. Conclusions and Future Work
    \- Summarize resolved limitations and extensibility (basis decomposition, advanced symbolic algebra, shared control extraction pass).

For every operation/modifier:
\- Provide: Purpose, Signature (ref + value semantics), Assembly Format, Builder Variants, Interface Implementation Notes, Canonicalization Rules, Examples (static + dynamic params), Conversion (ref↔value).
\- Ensure consistency of naming, capitalization, and parameter ordering.

Conventions:
\- Types capitalized (mqtref.Qubit).
\- Parameters in parentheses right after op mnemonic.
\- Use 2D tensor<2^n x 2^n> for matrices.
\- Do not reintroduce removed standalone sections (old passes, separate canonicalization inventory, etc.).
\- All canonicalization logic described inline; mention MLIR pattern categories (fold vs pattern).
\- Avoid speculative future work beyond concise Future Work subsection.

Deliverables:

1. Rewritten RFC text (single cohesive document).
2. No extraneous commentary; fully integrated narrative.
3. All examples syntactically valid MLIR.

Quality checklist before finalizing:
\- Section ordering matches specification.
\- No leftover references to UnitaryMatrix separate from UnitaryExpr.
\- All modifier examples updated to new nesting order.
\- cx not used as base gate example (only as sugar).
\- Block arguments correct for region wrappers (value semantics).
\- Canonicalization rules phrased as declarative transformations.

Now produce the full revised RFC accordingly.
