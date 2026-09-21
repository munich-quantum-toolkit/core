# CNOT–phase resynthesis

Status: complete.

## Goal and scope

The opt-in QCO pass `resynthesize-cnot-phase` optimizes bounded scalar regions
containing CNOT and diagonal single-qubit gates. General QCO programs remain
supported: other gates, tensor wire transfers, effects, and control flow delimit
regions. Independent disjoint operations need not delimit a region.

The implementation belongs in QCO transforms, with its public contract in
`mlir/include/mqt/Dialect/QCO/Transforms/Passes.td` and semantic tests in the
QCO optimization unit tests. No default pipeline or frontend basis changes.

## Decisions

- Independently implement GraySynth from Amy, Azimzadeh, and Mosca,
  [arXiv:1712.01859](https://arxiv.org/abs/1712.01859), with Gaussian
  elimination to restore the required final linear map. tzap provides
  engineering ideas, not implementation code.
- Retain each phase operation and parameter unchanged. This supports arbitrary
  and symbolic angles and preserves global phase without floating-point angle
  algebra or phase corrections. Phase merging and affine X propagation are
  outside this pass.
- Use 64-bit parity masks, bounded blocks, deterministic choices, and an early
  CNOT budget. Keep a candidate only when it reduces the CNOT count. Replay its
  parity network and final linear map before changing IR.
- Compute the residual map by replaying recorded CNOTs as column XORs on the
  desired map. This avoids a duplicate matrix and explicit matrix inversion;
  Gaussian elimination and the independent candidate checker remain.
- Insert at the region's end so captured angle definitions dominate all new
  uses. Verify QCO linearity at the pass boundary.

## Validation

The release optimization test executable passes all 218 tests. Its 17 new tests
include 150 deterministic generated circuits, full matrix comparisons, symbolic
and extreme angles, nested controls, tensor transfers, effects, resource limits,
and invalid input. Run from the repository root:

```sh
cmake --build --preset release --target mqt-core-mlir-unittest-optimizations
build/release/mlir/unittests/Dialect/QCO/Transforms/Optimizations/mqt-core-mlir-unittest-optimizations
```

The `mqt-cc` CLI also runs the registered pass successfully. Direct residual-map
construction produces identical output on all six benchmark circuits below.
`uvx nox -s cpp-lint` passes with no findings across all three changed C++
files. `uvx nox -s lint` passes for the full repository. The final diff contains
only the pass, its registration and tests, documentation, and a
spelling-dictionary entry for the C++20 bit-counting function.

## Measurements and limits

The three-gadget regression reduces eight CNOTs to four while keeping its three
phase gates. A release build on macOS arm64, using LLVM/MLIR 23.1.0 and
AppleClang 21, also processed six inputs from tzap revision
`605552533b9788aef60866266a4230318b6170a2`. The baseline imports QASM to QCO and
runs `decompose-multi-controlled,canonicalize,cse`; the comparison adds one
`resynthesize-cnot-phase` invocation with default limits.

| Circuit                      | Baseline CNOTs | After resynthesis |
| ---------------------------- | -------------: | ----------------: |
| `feynman/vbe_adder_3`        |             70 |                70 |
| `feynman/qcla_adder_10`      |            233 |               225 |
| `feynman/hwb6`               |            116 |               116 |
| `cobble-rz/laplacian-filter` |            660 |               660 |
| `cobble-rz/chebyshev`        |         20,459 |            19,748 |
| `cobble-rz/matrix-inversion` |         11,985 |            11,627 |

Five-run median CLI times on the same cleaned inputs were 318 ms versus 368 ms
for Chebyshev and 186 ms versus 213 ms for matrix inversion, comparing an empty
pipeline with resynthesis. These include process startup, parsing, verification,
and text output; they are not isolated pass times or execution-speed estimates.
The baselines differ from tzap's preprocessing, so these are not tool rankings.

Phase merging, affine X propagation, topology-aware synthesis, and PMH linear
resynthesis remain outside this pass. Native CZ and RZZ gates are boundaries;
the pass does not expand them to make a region. CNOT count improves on only
three of the six sampled circuits. Depth and routing cost are not optimized.
