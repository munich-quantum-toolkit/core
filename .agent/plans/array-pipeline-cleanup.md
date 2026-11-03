# Array pipeline cleanup

Status: complete; rebased onto the merged IQM export and locally validated.
Hosted CI is separate from local validation.

## Scope and decisions

Reduce repeated work in native QIR execution, frontend initialization analysis,
and program conversion/export. Keep the supported array subset and memref
representation. Compact literal initialization remains a separate follow-up.

- `Session.cpp` runs LLVM `MemCpyOptPass` before sampling's O1 pipeline and for
  aggregate loads in state extraction. Early scalar replacement otherwise
  expands bulk copies. Keep O1: a smaller pipeline loses useful loop
  optimization, while extra early DSE repeats expensive analysis. Exported QIR
  and target capabilities remain unchanged.
- `OpenQASMSemantics.cpp` visits selected initialization runs directly. Resolve
  every offset before mutation, check sources before targets, and detach target
  state once. Visit innermost strided elements without another recursive call;
  the first callback implementation regressed dense selections. The parser
  requires positive rank and each selection matches that rank; the innermost
  callback needs no scalar recursion leaf. Callers skip fully initialized
  storage, including empty arrays.
- `QCProgram::intoOpenQASM3()` and the owned low-level translator consume input,
  including on failure. Explicit copies use the existing `copy()` API or Python
  `copy=True`. Keep the context alive through diagnostics and module
  destruction. Translation still buffers output until success.
- QIR text and bitcode export consume owned MLIR before materializing native
  instruction names, eliminating a full-module clone. The canonical LLVM
  conversion used by IQM stays borrowed and read-only. C++ uses rvalue export
  methods; Python uses `to_llvm_ir`, `to_bitcode`, and `write_bitcode` with
  `copy=False` by default. Conversion failures consume ownership too.
- `compile_program` consumes typed MLIR after argument and target validation;
  `copy=True` preserves a source. Source strings, paths and external circuits
  remain borrowed. Python string input uses a view backed by its live immutable
  Python object. The QDMI adapter moves OpenQASM/IQM payloads and writes bitcode
  directly into its output buffer; Python bitcode avoids an intermediate vector.
- Remove 48 redundant explicit test copies, six builder/text/parser round trips,
  and two leaked raw module clones. Retain copies for independent variants,
  ownership-preservation contracts, and distinct text/bitcode artifacts.
  QIRRunner and DDSIM reuse the same serialized bytes. One focused regression
  checks source ownership on failed bitcode-file opening.
- Keep the jeff byte-import alignment copy and textual JIT parser's terminated
  buffer. Read-only serializers need no consuming wrapper; native symbol fanout
  still needs individual declaration clones. No serializer infrastructure is
  added.

## Validation

Existing frontend, compiler, exporter and Python tests cover initialization,
negative strides, branch isolation, ownership and round trips. Bypassing target
copy-on-write makes the branch-isolation test fail. The initial array cleanup
added no test cases. The rebased native release suite passes all 4,093 tests and
Python 3.14 passes all 2,035 tests. All ten QIRRunner cases pass after removing
unused text exports. The failed-write regression detects restored implicit
copying. C++ lint, regenerated stubs, repository lint, executable docs and
internal documentation links pass. Rendered QIR, ownership and Qiskit examples
were visually inspected, including console and network checks. A full Ponytail
review removed unreachable visitor paths and redundant test work. String/stream
equivalence and explicit inlining rejection still protect distinct contracts. No
correctness finding remains.

Fresh same-toolchain alternating comparisons verified checksums, sampled
results, statevectors and export bytes. A 1,000-element full-copy checksum
reduced warmed native JIT CPU from median 66.2 ms to 5.2 ms and optimized LLVM
from 4,020 to 20 instructions. Sparse selected-run analysis fell from 14.6 ms to
2.9 ms; public import fell from 78.3 ms to 66.7 ms. Dense cases overlap.
Consuming export removed 127.2 MB of allocation across three 50,000-gate
imports/exports; peak live heap stayed 63.03 MB while independent process peak
RSS fell. These are workload-specific results on a shared host. Phase timings
exclude file I/O and destruction; literal-QASM end-to-end ranges overlap. Raw
harnesses and measurements remain outside the repo.

A further aggregate-load gate for sampling showed no stable benefit and was
rejected. Keep the existing completeness fast paths and low-level verifier: new
caches or unchecked entry points lack a demonstrated benefit. Completeness
checks can still scan backing words; the whole analysis is not uniformly
proportional to selection size. Compact literal initialization remains deferred.

The ownership follow-up's matched QIR export experiment compiles both complete
translation units with identical Clang/LLVM settings. Seven alternating process
pairs export the same 10,005-native-call module produced from OpenQASM; LLVM
text is byte-identical. Export CPU medians are 29.753 versus 28.074 ms, with
paired ratios 0.9267–0.9610. Three parse/exports allocate 17.40 MB less and peak
live heap falls from 16.10 to 11.81 MB; independent process RSS medians are
32,608 versus 28,352 KiB. This isolates export, not whole-compiler throughput. A
specialist checked storage/context lifetimes and the merged IQM path. The review
removed an unused include and a redundant preservation-only test copy.
