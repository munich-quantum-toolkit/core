# Classical arrays during structured mapping

Status: complete.

## Outcome and scope

Numeric arrays imported from `jeff` now survive explicit-topology mapping with
runtime indexing, memory updates, aliases, and measurement feedback intact.
Opaque classical calls and unknown or value-less memory effects remain
unsupported and are diagnosed before mapping mutates the input.

## Decisions

`PromoteClassicalMemory` already uses upstream SROA and Mem2Reg before mapping.
Runtime-indexed arrays cannot always become SSA scalars. Retain their MemRef
storage and extend the existing classical dependency contract in
`mlir/lib/Dialect/QCO/Utils/Sorting.cpp` and the mapping pass to that storage.
Direct CBit and MemRef allocations own distinct storage. Other storage handles
may alias and therefore share a conservative effect chain. Such aliases also
require conservative measurement scheduling. For distinct allocations, routing
follows measurement feedback through storage users, including memory copies.

Mapping accepts MemRef operations only when their known effects identify the
storage values. The pass documentation states that boundary. The sorter still
does not order unknown or value-less effects; callers must exclude them.

## Validation

On main revision `04de176835af9dfc2c78f9f201cce932bfc0bc4d`, a two-element angle
array in a counted loop failed mapping. Repeating upstream promotion left its
runtime-indexed load intact. The added cases fail before the fix and pass
afterward. They verify the measured result, feedback through copies with and
without aliases, and store/load ordering through aliased handles. The existing
opaque-call failure-before-mutation regression still passes.

Build with the `release` preset and run the mapping, QCO utility, and compiler
unit-test executables. All 130 mapping and 255 utility tests pass. The compiler
suite passes 315 tests; two `/dev/full` tests skip on macOS. Repository lint
passes. Clang-tidy 23 reports no findings in the complete changed C++ files or
changed sorting header; generated and unchanged included headers emit warnings.

The structured-program coverage rerun spans all 16 registered benchmark families
through serialized exchange and QIR 2.1 Adaptive. All 25 all-to-all executions,
24 mapped executions, and mapped Shor compilation pass. The larger checks also
pass. Mapped Shor and level-2 distillation were checked through compilation
only; no runtime or scaling claim follows from these checks.
