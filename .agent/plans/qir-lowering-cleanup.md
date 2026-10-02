# QIR lowering cleanup

Status: complete.

## Scope and decisions

Preserve direct Base and Adaptive conversion passes and their supported input
contracts. Share their final classical lowering and cast reconciliation in
`QIRCommon`, while retaining profile-specific quantum lowering.

Remove the obsolete cleanup metadata rewrite and unused allocator state. Emit
computation-type lists as native LLVM module flags. LLVM 23 still requires the
integer flag-width repair after translation; direct i1/i2 flags can assert in
its integer translation path.

Allocate result-pointer vectors only in Base and move returned register records
into output emission in function-result order. Bind control operands to the
specific body gate instead of relying on rewrite order.

Use LLVM memset for local Adaptive register zero initialization at the original
allocation location. Compute the byte extent through an LLVM GEP so target
layout determines the i1 allocation stride.

Keep dynamic size support in the shared QIR builder utility. Defer symbol lookup
caching until profiling shows it matters. The broader resource, lifetime, and
profile-validation findings are outside this cleanup.

## Validation

Local validation passes: 288 Base and Adaptive conversion tests, 121 QIR
metadata/translation tests, 24 QIR JIT tests, and 171 compiler tests. The
conversion regressions check constant-size zero initialization for an 8,192-bit
register and initialization at a loop-local allocation site.

Build the corresponding release test targets, run
`ctest --preset release -R QCToQIR`, and run the QIR IR, QIR JIT, and compiler
test binaries. Repository lint and full-file C++ lint pass. Hosted CI is
separate from these local results.
