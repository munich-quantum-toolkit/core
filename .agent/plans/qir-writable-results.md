# QIR writable results

Status: complete.

## Outcome

The runtime and JIT execute the optional Adaptive Profile
`__quantum__rt__write_result(i1, ptr)` call for static and dynamic results.
Returned CBit registers use result slots for classical writes when their target
confirms support. Other targets retain Boolean storage.

## Decisions

- Select the extension through the existing target capability contract. Require
  complete optional capability metadata and unconstrained
  `qir.writable-results`; compiler maximum-capability lists alone do not prove
  backend support.
- Derive `writable_results` from runtime calls and validate it through the
  shared optional-capability table, as for dynamic allocation.
- Reuse checked result lookup and register-slot access. Overwriting a result
  clears its deferred wire without measuring or changing a qubit.
- Keep separately returned measurement snapshots in their own result slots.
  Preserve direct measurement/store fusion for other returned-register writes.

## Validation

- The release build and `mlir-doc` reference generation passed.
- `ctest --preset release --parallel 4`: 4,166 passed, 3 skipped on macOS. DDSIM
  worker tests require permission to bind local sockets.
- `uvx nox -s cpp-lint --force-python 3.14 -- HEAD` checked all 13 changed C++
  source and test files with clang-tidy 23.1.2. The final test-context edit also
  passed a separate full-file clang-tidy check.
- `uvx nox -s lint` passed. No Python bindings changed; Python tests and the
  full executable documentation build were not run.
