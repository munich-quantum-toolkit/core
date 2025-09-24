I am working on an MLIR dialect hierarchy for hybrid classical quantum computing. I have a working version of the two quantum dialects, but I am not happy because it is cumbersome to use and lack a lot of features.

The relevant files are contained solely in the `/Users/burgholzer/CLionProjects/mqt-core/mlir` directory.

The mlir-quantum-dialects.md file contains a detailed sketch of requirements for the envisioned architecture.

Take the existing setup as inspiration and think very hard and deep on how to replace it with a setup that facilitates the above requirements.
Nothing in the existing setup is set in stone.
There needs to be no migration path or backwards compatibility.
A fundamental rework is possible.

Do not (yet) perform any changes on the MLIR code itself. Rather provide an extensive implementation plan in a separate Markdown document.

---

I'd like to do one more iteration on the project plan before starting any edits. Address the following points

- QubitRegisters will be entirely replaced by using `memref` in a separate PR. The respective code here should not be touched as part of this refactor and the project plan might as well assume that QubitRegisters do not exist in both dialects
- I am unsure how uniform the UnitaryInterface can really be given how the dialects use different semantics and there would need to be a way to query input und output operands of the MQTOpt operations
- I like `UnitaryExpr` more than `UnitarySpec`. This description should really be as idiomatic as possible. I do not want to reinvent the wheel here. At the same time, I want to avoid further external dependencies. Most matrices will be 2*2 or 4*4 here and I need to be able to multiply them. It is very important that this is performant but lightweight. These matrices are truly only to be used for transformation or conversion passes.
- Instead of inv(pow(X, k)) -> pow(X, -k), perform inv(pow(X, k)) -> pow(inv(X), k) as a canonicalization
- Control SSA values should not be simply yielded as unchanged. Qubits should generally be treated as linear types wherever feasible
- The power modifier does not need a mask
- Perform pow(X, -k) -> pow(inv(X), k) as canonicalization
- Controls should be the outermost modifiers, inverse the innermost.
- Inverse modifiers can be canonicalized with base gates by changing the gate to the inverse. Each base gate has its inverse defined as a base gate (which might be the same gate but with different parameters)
- Allow even more syntactic sugar, e.g. mqtref.ccx %c0, %c1 %q
- Terminators of regions should be implicit and need not be explicit.
- The notion of controls and targets does not really make sense for `seq` operations. However, it makes sense to query the overall qubits a seq acts on.
- The UnitaryExpr of a `seq` operation is the product of the `UnitaryExpr` of the child operations.
- try to pick idiomatic names for the interface functions with regard to MLIR, C++ and Quantum Computing
- "Alternatively, keep a generic template that accepts targets as Span and parameters as ParamSet and dispatches based on trait arities; then layer nicer overloads on top." i like this solution
- make sure to always include examples in fenced mlir code blocks in the to be generated docstrings
- use canonicalization and folds as much as possible and idiomatic for MLIR. The goal should be to use as much of MLIR's native infrastructure as possible.
- there is no need to provide a temporary helper conversion. All code is expected to be rewritten and fixed

---

One last round of revisions and additions:

- I want to add an additional trait for "Hermitian" gates, i.e., gates that are self-inverse. These may be used as a short-circuit in inverse canonicalization. Popular hermitian operations are: I, X, Y, Z, H
- I want to add an additional trait for diagonal gates, i.e., gates whose underlying matrix is diagonal in the computational basis. Prominent examples include: I, Z, S, Sdg, T, Tdg, Rz, Phase (or P), RZZ
- parameters of base gate ops should be preferred to be static, constant folding (or another appropriate MLIR concpet) should be responsible for transforming dynamic operands that are constants to static attributes.
- do not put a limit on the arity for the constant DenseElementsAttr for the unitary. This will hold in practice by construction
- controls do not need deduplication as they must not be duplicated from the beginning, as also ensured by a verifier
- getQubitSet is redundant if `getAllOperandQubits()` and `getAllResultQubits()` are implemented properly. Qubits that are not acted on by a seq should not be part of the operands/results of the seq.

---

A couple of further requirements to factor into the plan:

- There should be extensive documentation on all the relevant design decisions that went into this project plan. This documentation should be present in the doxygen comments that will be included in the overall project documentation build. In addition, more elaborate rationale; especially on aspects that do not end up in the doxygen comments, should be added to the mlir docs infrastructrure under `docs/mlir`. The result should be a gentle introduction to all the necessary concepts (also including the ones not necessarily modified in this plan). The writing should be of a quality that could go into a (software focused) scientific journal. Always try to include examples in the documentation.
- Check the project plan for consistency and correct any ambiguity.
- I want to post the project plan as a high-level tracking issue on GitHub. Prepare it for that, e.g., replace all local paths with relative paths.
