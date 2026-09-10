# Compile structured benchmarks for devices

Status: complete.

## Scope and ownership

Make generated standard and iterative QPE and repeat-until-success programs
compile and execute through the device-directed API. Keep benchmark generators
unchanged. These compiler prerequisites are separate from the README refresh.

QIR lowering uses upstream MLIR bufferization and MemRef-to-LLVM conversion for
constant classical tensor reads. Other tensor operations remain unsupported. The
shared target loop-unrolling pass exposes fixed quantum tensor indices under its
existing expansion bound; supported scalar loops remain structured. QTensor
canonicalization lifts complete constant-index accesses out of a while loop and
preserves scalar state and the before-region result order. Tensor feedback must
return to the same iteration argument. Placement checks its tensor input
contract before changing allocations, including when called directly.

QIR output uses recording order. Benchmark evaluation uses big-endian strings;
the integration tests convert between these documented conventions.

## Validation

The generated iterative QPE, standard QPE, and one- and four-data-qubit RUS
programs compile to Adaptive QIR and execute on DDSIM. Exact QPE returns 3/8;
RUS results agree with the phase-sensitive analytic reference. Direct regression
tests cover constant table reads, tensor control-flow rejection, and while-loop
scalarization with static and runtime indices. All 658 relevant native tests and
27 device-compilation Python tests pass. General lint, full-file C++ lint,
strict HTML generation, and generated-link validation pass.
