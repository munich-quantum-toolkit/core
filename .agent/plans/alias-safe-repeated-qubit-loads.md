# Implement alias-safe repeated qubit loads

Status: historical implementation record.

## Goal and scope

MQT Core's QC builder currently rejects repeated loads of a qubit-register
element, and the typed OpenQASM frontend expands every runtime qubit index into
an `scf.index_switch` with one case per register element. After this change,
ordinary `memref.load` operations may name the same register element repeatedly,
including from the entry block, and QC-to-QCO conversion remains correct even
when two sequential runtime indices happen to alias. OpenQASM programs with
runtime qubit indices lower to a fixed-size load/use sequence instead of code
whose size grows with the register width.

The observable proof is twofold. A raw QC module containing repeated qubit loads
converts to verified QCO in which each register operand is extracted immediately
before its quantum operation and inserted immediately afterward. An OpenQASM
program containing `x q[i];` emits a checked `memref.load` and no
width-dependent `scf.index_switch`.

## Constraints

- MLIR's LLVM 22 CSE cannot be the correctness mechanism. QC quantum operations
  declare memory writes, so an intervening gate prevents CSE from reusing a
  prior `memref.load`.

- The current OpenQASM frontend no longer emits dynamic loads. It selects
  eagerly loaded references through nested `scf.index_switch` operations, making
  emitted operation count depend on register widths.

- Existing QTensor canonicalizers scan only constant-index tensor chains.
  Dynamic indices can alias unless they are the exact same SSA value, so only
  adjacent exact-index folds are safe without an alias analysis.

- The old tensor-chain search can cross a structured QCO operation. Once
  structured regions carry complete tensors, folding an insert against an
  extract on the far side of that operation bypasses region-local quantum
  updates. The replacement canonicalizers therefore use direct SSA
  producer/consumer relationships only.

- `ValueRange{value}` does not own its initializer-list storage. Keeping such a
  range in a local variable produced a dangling view during the first conversion
  prototype. Single-qubit helper calls now use owned `SmallVector<Value, 1>`
  storage.

- Once quantum dispatch is removed, the existing emission budget correctly
  accepts registers that were previously rejected solely because their widths
  were multiplied into projected operation counts. Large and small registers now
  produce identical operation counts for the same dynamic source access.

- The QCO mapping pass intentionally expects a canonical
  all-extracts-before-all-inserts tensor shape. The public place-and-route API
  now composes the existing QCO cleanup pipeline before mapping, which restores
  that supported shape for statically addressable programs without weakening the
  operation-local conversion invariant.

- MLIR 22's canonicalizer defaults to ten outer greedy iterations. Commuting an
  insertion through an arbitrarily long unrolled register-access chain may
  require more iterations, leaving 16- and 100-qubit mapping inputs only
  partially normalized. Requesting convergence also exposed that the existing
  allocation/deallocation fold did not check for other allocation uses before
  erasing the allocation.

- Several direct-QCO test fixtures encoded the old partially extracted
  structured state. Rewriting those fixtures to allocate complete QTensors
  directly exposed and verified the new region-boundary invariant while
  preserving direct QC-to-QIR behavior.

- The aggregate lint cache contained incomplete hook environments from an
  interrupted provisioning run. Moving only the generated cache aside and
  allowing `prek` to recreate it produced a clean all-files lint run.

- Gates already rejected exact duplicate qubits during semantic analysis and
  asserted potentially aliasing dynamic operands at runtime, but barriers
  originally did neither. Reusing the runtime assertion path for barriers
  preserves the linear QCO contract. Grouping accesses by register and using
  LLVM dense containers avoids quadratic scans over expanded whole registers
  when all indices are statically distinct.

- MLIR already provides `getUsedValuesDefinedAbove` for region capture
  discovery. Using it for each supported SCF operation avoids recursively
  interpreting block locality and correctly distinguishes values captured from
  enclosing regions from qubits allocated inside a structured region.

- The QC modifier verifiers already reject register loads inside modifier
  bodies. Register-specific modifier maps were therefore dead state; modifier
  operands need only the existing standalone SSA alias mapping.

- The legacy OpenQASM translation reference test repaired eager register
  fixtures with a custom, order-sensitive IR mutation. Lowering both modules
  through the production QC-to-QCO pass yields a stronger semantic comparison
  and deletes that bespoke normalizer.

- `QCOProgramBuilder::qtensorAlloc(1)` and
  `QCOProgramBuilder::allocQubitRegister(1)` have intentionally different
  linear-state results. The first returns one intact tensor, while the second
  eagerly extracts its element and returns the residual tensor plus a standalone
  qubit. The corresponding QC APIs now document the same storage-only versus
  eager-reference distinction.

- QC modifier verification previously allowed a qubit from an enclosing region
  to be used without appearing among the modifier operands. QC-to-QCO maps only
  the aliased modifier block arguments, so such a capture reached an assertion
  when pass verification was disabled. MLIR `getUsedValuesDefinedAbove` provides
  the exact capture query needed by both the verifier and conversion preflight
  while leaving classical captures valid.

- Collecting load provenance must not assume that the source memref is rank one.
  Rank-zero `memref<!qc.qubit>` is valid MLIR, and calling `front()` on its
  empty index range asserted. The preflight now validates storage shape and
  value origin before recording provenance.

- `BaseMemRefType`, rather than `MemRefType`, is required when classifying
  unsupported quantum storage. Otherwise unranked `memref<*x!qc.qubit>` block
  arguments and operations can remain dynamically legal and escape conversion
  unchanged.

- Only `scf.for`, `scf.while`, `scf.if`, and `scf.index_switch` participate in
  QC-to-QCO's explicit quantum-state threading. Capturing quantum values through
  another region-bearing operation, such as `scf.execute_region`, must be
  diagnosed before rewriting; purely classical uses of those operations remain
  legal.

- QIR Base intentionally requires single-block straight-line QC input and does
  not lower SCF. Dynamic loop programs therefore target the Adaptive profile;
  the Base-profile dynamic-loop failure is not caused by this change, and the
  QC-to-QIR implementations remain untouched.

- The nested `scf.for`/`scf.while` conversion test originally skipped reference
  equivalence because the QTensor iterator did not recognize `scf.condition` as
  a linear-chain terminator and IR equivalence did not assign the loop's before-
  and after-region tensor values to the allocation's equivalence group. Modeling
  those standard SCF edges removes the test exception and lets the existing
  permutation-aware oracle compare the complete structured state.

- QCO-to-QC intentionally exercises the older partially extracted QTensor
  fixture, whereas Q-01 needs a complete-QTensor QC-to-QCO reference. These are
  distinct conversion inputs rather than interchangeable names, so both test
  builders must remain available.

## Decisions

- Treat qubit `memref.load` as reference provenance and materialize QCO values
  around each consuming quantum operation. Rationale: This supports arbitrary
  sequential runtime aliasing without adding an analysis, dialect type, or
  operation.

- Keep simultaneous operands subject to the existing distinct-qubit contract and
  preserve OpenQASM runtime assertions. Rationale: A multi-qubit QCO operation
  cannot extract the same linear qubit twice.

- Preserve the public eager `allocQubitRegister` API and add a storage-only
  primitive for the frontend. Rationale: Existing builders and fixtures remain
  source-compatible.

- Preserve exact QCO reference comparison where representation is unchanged, and
  assert operation-local extract/insert linearity plus complete tensor state for
  structured-register cases. Rationale: Eagerly extracted QCO references encode
  the representation Stage 1 intentionally removes and are no longer valid
  structural references for those cases.

- Treat barriers as simultaneous multi-qubit operations for alias validation.
  Rationale: A QCO barrier consumes its operands linearly just like a
  multi-qubit gate; exact duplicates should fail in semantic analysis, while
  potentially equal runtime indices need `cf.assert`.

- Use MLIR `RegionUtils` to discover structured captures and keep the
  conversion's custom logic limited to classifying captured quantum values as
  standalone qubits or register provenance. Rationale: Region ownership and
  nested block locality are standard MLIR concerns and should not be
  reimplemented recursively.

- Reject only simultaneous register operands whose equality is statically proven
  by identical SSA values or equal integer constants. Rationale: QCO linearity
  makes those operations invalid, while distinct dynamic values may alias only
  at runtime and remain the source frontend's responsibility to guard.

- Keep `qtensorAlloc`/storage-only allocation distinct from `allocQubitRegister`
  eager extraction, but make the distinction explicit in public documentation
  and use storage-only APIs whenever complete tensor state is required.
  Rationale: Collapsing the APIs would either hide linear extraction or break
  source compatibility.

- Reject implicit qubit captures in QC modifiers at both operation verification
  and conversion preflight. Rationale: Modifier bodies may capture classical
  parameters, but every quantum value must enter through an aliased modifier
  block argument so the QCO linear mapping is complete.

- Centralize the supported QC quantum-value-source contract in a read-only
  preflight using MLIR `BaseMemRefType` and `getUsedValuesDefinedAbove`.
  Rationale: The lowering state can map only `qc.alloc`, `qc.static`, direct
  rank-one register loads, QC modifier qubit arguments, and captures through the
  four explicitly converted SCF operations; diagnosing every other source before
  dialect conversion prevents assertion and dominance failures without adding
  alias analysis.

- After the complete release build, CTest matrix, hooks, and one Python session
  passed, limit the final iterations to the affected conversion binary,
  changed-line clang-tidy, and compiler smokes. Rationale: The user explicitly
  requested shorter iterations once the broad validation was working.

- Keep both pre- and post-mapping QCO cleanup in target compilation. Rationale:
  The existing cleanup canonicalizes operation-local QTensor access into the
  mapper's supported form without adding a bespoke transformation, while the
  later cleanup preserves established post-mapping normalization.

- Let the QCO cleanup canonicalizer run to convergence and preserve the standard
  MLIR greedy driver rather than adding a tensor-size-dependent pass loop or a
  larger arbitrary cap. Rationale: complete QTensor normalization is a mapping
  precondition and OpenQASM registers have no fixed width limit; QTensor rewrite
  patterns are local and monotonic. Guard the pre-existing
  allocation/deallocation rewrite with `hasOneUse()` so the stronger cleanup
  cannot erase a still-used allocation.

- Close Q-01 through the existing permutation-aware equivalence infrastructure
  rather than adding a one-off comparison path. Rationale: `scf.condition` and
  the two `scf.while` regions are standard parts of the QTensor def-use graph,
  and supporting them centrally strengthens every equivalence user while keeping
  the production conversion unchanged.

## Outcome and validation

Quantum memref loads carry reference provenance; each quantum use lowers to a
local extract/use/reverse-insert sequence. Structured regions carry complete
tensor state. The frontend emits checked point-of-use loads without register-
width-dependent switch expansion.

Shared preflight rejects unsupported quantum sources and implicit modifier
captures before mutation. Gates and barriers reject proven simultaneous aliases;
runtime checks cover only pairs that may alias. A 10,000-element static barrier
needs no runtime alias checks. MLIR RegionUtils supplies capture discovery.

Historical validation passed the release build, configured CTests with two
existing skips, repository hooks, the focused conversion suite, and QC/QCO/QIR
compiler smokes. Python 3.10 passed; the remaining interpreter repetitions were
not run. Changed-line clang-tidy was supplemental validation.

## Code and ownership

The QC dialect models `!qc.qubit` as a reference. A QC gate mutates that
reference in place. The QCO dialect models `!qco.qubit` linearly: an operation
consumes an input SSA value and produces the next value. A QTensor is a
one-dimensional `tensor<...x!qco.qubit>` whose `qtensor.extract` removes one
qubit and whose `qtensor.insert` returns it.

`mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` and
`mlir/lib/Dialect/QC/Builder/QCProgramBuilder.cpp` own the public builder.
Before this work, `allocQubitRegister` allocated a qubit memref and eagerly
loaded every constant element, while `loadQubit` rejected entry-block and
repeated loads using per-region maps.

`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` converts QC reference semantics to QCO
value semantics. Its previous load pattern extracted a qubit when the load was
converted and kept it live until a structured boundary or register deallocation.
That was unsafe when another load could name the same element.

`mlir/lib/Dialect/QTensor/IR/Operations/ExtractOp.cpp` and `InsertOp.cpp` own
local tensor-chain canonicalization. The former implementations searched
constant-index chains; the replacements fold only direct producer/consumer pairs
whose indices are equal by MLIR's standard value-or-constant utility.

`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` emits QC from the
typed OpenQASM frontend. Before Stage 2, non-scalar declarations retained
vectors of eagerly loaded values and dynamic references recursively constructed
`scf.index_switch` operations over those vectors.

Tests belong under `mlir/unittests/Conversion/QCToQCO/`,
`mlir/unittests/Dialect/QC/IR/`, `mlir/unittests/Dialect/QTensor/IR/`, and
`mlir/unittests/Dialect/QC/Translation/`. Production tools are not test
locations.

## Acceptance

Stage 1 is accepted when builder tests can emit repeated identical loads in the
entry block and nested SCF, and raw `createQCToQCO()` conversion verifies
without first running canonicalization or CSE. Tests must cover constant and
dynamic indices, repeated sequential use, an eager constant reference followed
by a possibly aliasing dynamic reference, structured control flow, measurement,
reset, barriers, and modifiers. The resulting QCO must contain complete tensors
at structured boundaries and register deallocation.

QTensor canonicalization is accepted when adjacent operations at the same
constant or exact dynamic SSA index fold, while adjacent operations at distinct
dynamic SSA indices remain unchanged.

Stage 2 is accepted when OpenQASM dynamic quantum operations emit direct
register loads and no quantum-selection `scf.index_switch`. A large register
must emit approximately the same number of operations as a small register for
the same source operation. Existing index bounds, negative-index, and
same-qubit-operand failures must remain observable.

The complete change is accepted when focused tests, affected compiler tests, the
release build, repository lint, `git diff --check`, and independent review pass.
Any environment-limited check must be recorded with exact evidence rather than
weakened.

## Interfaces

The only new public C++ interface is:

    Value QCProgramBuilder::allocQubitRegisterStorage(int64_t size);

`QCProgramBuilder::allocQubitRegister(int64_t)` remains source-compatible.
`QCProgramBuilder::loadQubit(Value memref, Value index)` retains its signature
but permits repeated and entry-region loads.

The implementation uses existing MLIR 22 APIs: `memref::LoadOp`,
`qtensor::ExtractOp`, `qtensor::InsertOp`, `isEqualConstantIntOrValue`, dialect
conversion patterns, SCF iter arguments, and the repository's existing LLVM
containers. It adds no external dependency, dialect operation, dialect type,
pass, or command-line option.
