# MLIR Quantum Compilation Infrastructure Implementation Prompt

You are an expert MLIR compiler developer tasked with implementing a comprehensive MLIR-based quantum compilation infrastructure for the Munich Quantum Toolkit (MQT). Your implementation will follow the detailed technical specification provided in the `docs/mlir/mlir-compilation-infrastructure-technical-concept.md` document.

## Project Context

**Location:** The implementation lives in the top-level `mlir/` directory of the MQT Core project.

**Approach:** You should build on top of the existing infrastructure where beneficial, but **backwards compatibility is NOT a concern**. Full refactors are preferred if they better achieve the design goals. The existing code can serve as inspiration but should not constrain the implementation.

**Build Configuration:**

```bash
cmake -S . -B build-mlir-copilot \
  -DLLVM_EXTERNAL_LIT=/home/lburgholzer/Documents/mqt-core/.venv/bin/lit \
  -DBUILD_MQT_CORE_MLIR=ON \
  -DMLIR_DIR=/usr/lib/llvm-21/lib/cmake/mlir \
  --config Release
```

**Build Targets:**

- Build MLIR tests: `cmake --build build-mlir-copilot --target mqt-core-mlir-lit-test-build-only`
- Run MLIR lit tests: `cmake --build build-mlir-copilot --target mqt-core-mlir-lit-test`
- Build translation tests: `cmake --build build-mlir-copilot --target mqt-core-mlir-translation-test`
- Run translation tests: `./build-mlir-copilot/mlir/unittests/translation/mqt-core-mlir-translation-test`

**Code Quality:**

- Lint the project: `pre-commit run -a`

## Implementation Phases

Your implementation should proceed through the following phases, implementing the design systematically:

### Phase 1: Core Type System and Dialect Infrastructure

**Goal:** Establish the foundation for both Quartz and Flux dialects.

**Tasks:**

1. Define `!quartz.qubit` and `!flux.qubit` types in their respective dialects
2. Implement the `UnitaryOpInterface` with all methods specified in Section 3.5:

- Qubit accessors (targets, positive/negative controls)
- Value semantics threading (inputs/outputs)
- Parameter handling with `ParameterDescriptor`
- Matrix extraction (`hasStaticUnitary()`, `tryGetStaticMatrix()`)
- Modifier state (`isInverted()`, `getPower()`)
- Identification methods (`getBaseSymbol()`, `getCanonicalDescriptor()`)

3. Set up dialect infrastructure with proper CMake integration
4. Implement basic verification framework

**Acceptance Criteria:**

- Both dialects register successfully with MLIR
- `UnitaryOpInterface` compiles and can be attached to operations
- CMake build succeeds without errors

### Phase 2: Resource and Measurement Operations

**Goal:** Implement non-unitary operations for qubit management and measurement.

**Tasks:**

1. **Resource Operations (Section 3.3):**

- `quartz.alloc` / `flux.alloc`
- `quartz.dealloc` / `flux.dealloc`
- `quartz.qubit` / `flux.qubit` (static qubit references)
- Integration with `memref` for quantum and classical registers

2. **Measurement and Reset (Section 3.4):**

- `quartz.measure` / `flux.measure` (computational basis only)
- `quartz.reset` / `flux.reset`
- Proper type signatures for both reference and value semantics

3. Implement canonicalization patterns for these operations
4. Add verification rules

**Acceptance Criteria:**

- All resource operations parse, verify, and print correctly
- Canonicalization patterns trigger appropriately
- Unit tests validate each operation's behavior

### Phase 3: Base Gate Operations - Part 1 (Zero/Single-Qubit Gates)

**Goal:** Implement all single-qubit base gates with full interface support.

**Tasks:**

1. Implement gates from Section 4.3:

- **Zero-qubit:** `gphase` (4.3.1)
- **Pauli gates:** `id` (4.3.2), `x` (4.3.3), `y` (4.3.4), `z` (4.3.5)
- **Fixed gates:** `h` (4.3.6), `s` (4.3.7), `sdg` (4.3.8), `t` (4.3.9), `tdg` (4.3.10), `sx` (4.3.11), `sxdg` (4.3.12)
- **Rotation gates:** `rx` (4.3.13), `ry` (4.3.14), `rz` (4.3.15), `p` (4.3.16), `r` (4.3.17)
- **Universal gates:** `u` (4.3.18), `u2` (4.3.19)

2. For each gate:

- Implement proper traits (`OneTarget`, `NoParameter`, `Hermitian`, etc.)
- Implement `UnitaryOpInterface` methods
- Define static matrix representation where applicable
- Implement all specified canonicalization patterns

3. Support both static (attribute) and dynamic (SSA value) parameters
4. Add comprehensive unit tests for each gate

**Acceptance Criteria:**

- All single-qubit gates parse and verify correctly in both dialects
- Static matrix extraction works for all parameter-free gates
- Dynamic matrix computation works for parameterized gates with static parameters
- All canonicalization patterns trigger correctly
- Global phase preservation semantics are respected

### Phase 4: Base Gate Operations - Part 2 (Two-Qubit Gates)

**Goal:** Implement all two-qubit base gates.

**Tasks:**

1. Implement gates from Section 4.3:

- **Entangling gates:** `swap` (4.3.20), `iswap` (4.3.21), `dcx` (4.3.22), `ecr` (4.3.23)
- **Ising-type gates:** `rxx` (4.3.24), `ryy` (4.3.25), `rzx` (4.3.26), `rzz` (4.3.27)
- **General gates:** `xx_plus_yy` (4.3.28), `xx_minus_yy` (4.3.29)
- **Utility gates:** `barrier` (4.3.30)

2. For each gate:

- Implement proper traits (`TwoTarget`, parameter arities)
- Implement `UnitaryOpInterface` with 4×4 matrices
- Implement canonicalization patterns

3. Special handling for `barrier`:

- Implements `UnitaryOpInterface` but acts as compiler directive
- Prevents optimization reordering
- Scope-based global phase aggregation

**Acceptance Criteria:**

- All two-qubit gates verify and print correctly
- 4×4 matrix extraction works for all gates
- Canonicalization patterns function correctly
- Barrier semantics are properly enforced

### Phase 5: Modifier Operations

**Goal:** Implement the complete modifier system for composable gate transformations.

**Tasks:**

1. **Control Modifiers (Section 5.2):**

- Implement `quartz.ctrl` / `flux.ctrl`
- Implement `quartz.negctrl` / `flux.negctrl`
- Support arbitrary nesting and flattening
- Implement control qubit exclusivity verification
- Handle value threading in Flux dialect

2. **Inverse Modifier (Section 5.3):**

- Implement `quartz.inv` / `flux.inv`
- Double inverse cancellation
- Hermitian gate simplification
- Parametric rotation inversion

3. **Power Modifier (Section 5.4):**

- Implement `quartz.pow` / `flux.pow`
- Support static and dynamic exponents
- Handle negative powers (convert to `inv(pow(+exp))`)
- Integer and real/rational exponent handling
- Matrix exponentiation via eigendecomposition

4. **Canonical Ordering:**

- Implement canonicalization rules to enforce `negctrl → ctrl → pow → inv` ordering
- Nested modifier flattening

5. **Unitary Computation:**

- Extend matrices to control spaces
- Conjugate transpose for inversion
- Matrix exponentiation for powers

**Acceptance Criteria:**

- All modifiers parse, verify, and execute correctly
- Canonical ordering is enforced through canonicalization
- Nested modifiers flatten appropriately
- Control modifier exclusivity is verified (no qubit in both `ctrl` and `negctrl`)
- Matrix extraction works through modifier layers
- Unit tests cover all canonicalization patterns

### Phase 6: Box Operation (Scoped Sequences)

**Goal:** Implement the `box` operation for scoped unitary composition.

**Tasks:**

1. Implement `quartz.box` / `flux.box` (Section 6)
2. Handle value threading for Flux dialect (region arguments and yields)
3. Implement canonicalization:

- Empty box elimination
- Single-operation inlining
- Nested box flattening

4. Composite unitary computation (product of constituent unitaries)
5. Verification rules

**Acceptance Criteria:**

- Box operations construct and verify correctly in both dialects
- Flux dialect properly threads values through region
- Canonicalization patterns trigger appropriately
- Composite matrix multiplication works for static sequences

### Phase 7: Builder APIs

**Goal:** Provide ergonomic programmatic APIs for circuit construction.

**Tasks:**

1. **QuartzProgramBuilder (Section 8.2):**

- Resource management methods
- All base gate methods (single and two-qubit)
- Convenience methods (`cx`, `mcx`, multi-controlled gates)
- Modifier methods (lambdas for body construction)
- Box scoping
- Gate definition and application placeholders
- Finalization with optional default pass application

2. **FluxProgramBuilder (Section 8.3):**

- Similar API but with SSA value returns
- Proper value threading
- Tuple returns for multi-qubit gates

3. **Integration:**

- Initialize MLIR context
- Create module and function
- Provide fluent chaining interface
- Apply default passes on finalization (optional)

**Acceptance Criteria:**

- Both builders construct valid MLIR modules
- Fluent chaining works correctly
- Generated IR matches hand-written IR structurally
- Example circuits (Bell state, GHZ) build successfully

### Phase 8: Testing Infrastructure Refactor

**Goal:** Establish robust, maintainable testing following Section 9.

**Tasks:**

1. **Unit Testing Framework (Section 9.2):**

- Set up GoogleTest infrastructure
- Test categories:
  - Operation construction
  - Canonicalization patterns
  - Interface implementation
  - Matrix extraction
  - Modifier composition
  - Dialect conversion
- Structural equivalence checks (not SSA name matching)
- Numerical matrix comparison with tolerances

2. **Minimal FileCheck Tests (Section 9.3):**

- Parser/printer round-trip validation
- Focus on structural properties, not formatting
- Avoid brittle SSA name checks

3. **Integration Tests (Section 9.4):**

- Frontend translation tests (Qiskit, OpenQASM 3, `qc::QuantumComputation`)
- Compiler pipeline tests
- Backend translation tests (QIR)
- End-to-end algorithm scenarios

4. **Default Pass Pipeline (Section 9.5):**

- Integrate `-canonicalize` and `-remove-dead-values` as defaults
- Configure builder API to optionally apply on finalization
- Document testing patterns using default pipeline

**Acceptance Criteria:**

- Comprehensive unit test coverage for all operations
- FileCheck tests are minimal and focused
- Integration tests cover key scenarios
- All tests pass reliably
- Test execution time is reasonable (<5 minutes for full suite)

### Phase 9: User-Defined Gates (Deferred)

**Note:** Section 7 explicitly defers detailed specification of user-defined gates. This phase is a placeholder for future work once core infrastructure stabilizes.

**Future Tasks:**

- Design concrete syntax for `define_matrix_gate` and `define_composite_gate`
- Implement symbol table integration
- Define verification rules for unitarity and consistency
- Create `apply` operation for gate instantiation
- Integrate with modifier system
- Design builder API convenience methods

## Implementation Guidelines

### Code Organization

```
mlir/
├── include/
│   └── mqt/
│       └── Dialect/
│           ├── Quartz/
│           │   ├── IR/
│           │   │   ├── QuartzDialect.h
│           │   │   ├── QuartzOps.h
│           │   │   ├── QuartzTypes.h
│           │   │   └── QuartzInterfaces.h
│           │   └── Transforms/
│           │       └── Passes.h
│           └── Flux/
│               ├── IR/
│               │   ├── FluxDialect.h
│               │   ├── FluxOps.h
│               │   ├── FluxTypes.h
│               │   └── FluxInterfaces.h
│               └── Transforms/
│                   └── Passes.h
├── lib/
│   └── Dialect/
│       ├── Quartz/
│       │   ├── IR/
│       │   │   ├── QuartzDialect.cpp
│       │   │   ├── QuartzOps.cpp
│       │   │   └── QuartzTypes.cpp
│       │   └── Transforms/
│       │       └── Canonicalize.cpp
│       └── Flux/
│           ├── IR/
│           │   ├── FluxDialect.cpp
│           │   ├── FluxOps.cpp
│           │   └── FluxTypes.cpp
│           └── Transforms/
│               └── Canonicalize.cpp
├── tools/
│   └── mqt-opt/
│       └── mqt-opt.cpp
├── test/
│   ├── Dialect/
│   │   ├── Quartz/
│   │   └── Flux/
│   └── Integration/
└── unittests/
    ├── Dialect/
    │   ├── Quartz/
    │   └── Flux/
    └── Builder/
```

### TableGen Usage

- Define operations using ODS (Operation Definition Specification)
- Define interfaces using TableGen interface definitions
- Use traits to express operation properties
- Generate boilerplate code where possible

### Naming Conventions

- **Dialects:** `quartz`, `flux` (lowercase)
- **Operations:** `quartz.h`, `flux.ctrl`, etc. (lowercase, dialect prefix)
- **Types:** `!quartz.qubit`, `!flux.qubit`
- **C++ Classes:** `QuartzDialect`, `HadamardOp`, `UnitaryOpInterface` (PascalCase)
- **Methods:** `getNumTargets()`, `tryGetStaticMatrix()` (camelCase)

### Verification Philosophy

- Verify type correctness
- Verify qubit uniqueness constraints
- Verify control/target disjointness
- Verify region structure (single block, proper terminator)
- Verify interface implementation consistency
- **Do not** verify unitary correctness (too expensive, user responsibility)

### Canonicalization Strategy

- Implement patterns as separate classes inheriting from `OpRewritePattern`
- Register patterns in `getCanonicalizationPatterns()`
- Patterns should be confluent (order-independent)
- Use pattern benefit values to prioritize important patterns
- Document pattern purpose and examples

### Matrix Computation

- Use `mlir::DenseElementsAttr` for static matrices
- Store as `tensor<NxNxcomplex<f64>>` where N = 2^(num_qubits)
- Use numerical libraries (Eigen, if available) for:
  - Matrix multiplication
  - Eigendecomposition
  - Exponentiation
- Handle numerical precision: machine epsilon for comparisons
- Return `std::nullopt` for symbolic/dynamic cases

### Testing Best Practices

1. **Unit Tests:**

- One test fixture per operation type
- Test construction with valid and invalid parameters
- Test each canonicalization pattern independently
- Test interface methods return correct values
- Use builders for construction, not raw IR strings

2. **Integration Tests:**

- Use realistic circuit examples
- Verify unitary equivalence numerically
- Test complete compilation pipelines
- Include performance benchmarks for large circuits

3. **FileCheck Tests:**

- Minimal use, only for round-trip parsing
- Check operation presence, not SSA names
- Use `CHECK-LABEL` for test separation
- Keep tests small and focused

## Implementation Checklist

For each phase, ensure:

- [ ] Code compiles without warnings
- [ ] All operations have TableGen definitions
- [ ] All operations implement required interfaces
- [ ] Verification logic is comprehensive
- [ ] Canonicalization patterns are implemented and tested
- [ ] Unit tests achieve high coverage
- [ ] Documentation comments are complete
- [ ] Code passes `pre-commit run -a`
- [ ] CMake integration is correct
- [ ] All specified tests pass

## Key Design Principles to Preserve

1. **Separation of Concerns:** Base gates are modifier-free; use wrappers for extensions
2. **Canonical Forms:** Enforce consistent representation through canonicalization
3. **Interface Uniformity:** All unitary operations expose the same introspection API
4. **Semantic Clarity:** Reference semantics (Quartz) vs. value semantics (Flux) are distinct
5. **Extensibility:** Design accommodates future dialects and abstraction levels
6. **Global Phase Preservation:** Treat global phases carefully; only eliminate via explicit passes
7. **Numerical Robustness:** Use appropriate tolerances and stable algorithms

## Success Criteria

The implementation is successful when:

1. All operations in Sections 3-6 are fully implemented and tested
2. Both Quartz and Flux dialects function correctly with proper semantics
3. Builder APIs enable ergonomic circuit construction
4. Testing infrastructure provides confidence through structural and semantic validation
5. Default pass pipeline normalizes IR consistently
6. Documentation is complete and accurate
7. Code quality meets project standards (linting passes)
8. All build targets succeed
9. Integration tests demonstrate end-to-end functionality

## Getting Started

**Step 1:** Read the complete specification in `docs/mlir/mlir-compilation-infrastructure-technical-concept.md`

**Step 2:** Set up your build environment using the configuration command

**Step 3:** Explore the existing `mlir/` directory to understand current structure

**Step 4:** Begin with Phase 1, implementing core types and interfaces

**Step 5:** Incrementally build up functionality, testing at each stage

**Step 6:** Use the existing translation tests as inspiration for test structure

**Step 7:** Iterate on design decisions, documenting deviations from the spec

## Questions and Clarifications

If you encounter ambiguities or need clarifications:

1. Check if the specification explicitly defers the decision (e.g., user-defined gates)
2. Look for related patterns in existing MLIR dialects (e.g., `arith`, `tensor`, `scf`)
3. Choose the simplest correct implementation that satisfies the requirements
4. Document your decision for future reference

## Deliverables

At the end of each phase, provide:

1. **Code:** Fully implemented operations, types, interfaces, and passes
2. **Tests:** Comprehensive unit tests and targeted integration tests
3. **Documentation:** Inline code comments and any necessary updates to the specification
4. **Build Validation:** Confirmation that all build and test targets pass
5. **Summary:** Brief description of what was implemented and any notable decisions

---

**Remember:** This is a comprehensive implementation project that requires attention to detail, adherence to MLIR best practices, and systematic testing. Focus on correctness and clarity over premature optimization. The goal is to create a robust foundation for quantum compilation that can evolve with the field.
