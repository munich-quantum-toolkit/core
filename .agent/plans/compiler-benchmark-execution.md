# Compile structured benchmarks for devices

Status: complete.

## Scope and ownership

Make generated standard and iterative QPE and repeat-until-success programs
compile and execute through the device-directed API. Keep benchmark generators
unchanged. These compiler prerequisites are separate from the README refresh.

QIR lowering uses upstream MLIR bufferization and MemRef-to-LLVM conversion for
constant classical tensor reads. Other tensor operations remain unsupported. The
target pipeline cleans up structured control flow before specializing loops.
Indexed register placement and shared if/for/while scalarization are documented
in [the follow-up plan](compiler-placement.md). Placement still diagnoses
unsupported tensor input before changing allocations, including when called
directly. The same follow-up diagnoses unsupported partial dynamic tensor
ownership and normalizes DDSIM QIR result bit order.

The QIR runtime preserves recording order. DDSIM normalizes that order when
returning QDMI shots and counts, so benchmark evaluation consumes job results
directly.

## Validation

The generated iterative QPE, standard QPE, and one- and four-data-qubit RUS
programs compile to Adaptive QIR and execute on DDSIM. Exact QPE returns 3/8;
RUS results agree with the phase-sensitive analytic reference. Direct regression
tests cover constant table reads, tensor control-flow rejection, and while-loop
scalarization with static and runtime indices. All 658 relevant native tests and
27 device-compilation Python tests pass. General lint, full-file C++ lint,
strict HTML generation, and generated-link validation pass.
