# Frozen routing fixtures

These artifacts were emitted by MQT Core
`1d384a5d23aa1f02b26fd5684dab586ef26167fe` from `asymmetric.qasm`. They
deliberately retain the compiler's output, including global phase, original
source tags, layout dictionaries, sparse site identifiers, and readout
destinations. Tests must compare against the source circuit without rerunning
the transformation under test to normalize both programs.

The common compiler options were:

```text
mqt-cc asymmetric.qasm --emit=qco-optimized --seed=42 \
  --mapping-trials=1 --mapping-iterations=0 \
  --target='<target below>' \
  --payload-spec='#mqt.payload_spec<format = <id = "openqasm", version = "3.1.0", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>' -o -
```

Each target has explicit connectivity and native operations `u` (arity 1, three
parameters), `cx` (arity 2), `gphase` (arity 0, one parameter), and `measure`
(arity 1). The exact normalized target attribute is retained in each emitted
module's `mqt.target_env`.

| Fixture               | Sites, edges, and compiler stage                                         |
| --------------------- | ------------------------------------------------------------------------ |
| `before-elision.mlir` | Sites 0, 1, 2; edges 0–1–2; immediately before terminal SWAP elision     |
| `after-elision.mlir`  | Same target; immediately after terminal SWAP elision                     |
| `sparse-sites.mlir`   | Sites 10, 30, 20; edges 10–30–20; final optimized QCO                    |
| `workspace.mlir`      | Sites 0, 1, 2, 3; edges 0–1–2–3; final optimized QCO with idle workspace |
| `measured.mlir`       | Sites 0, 1, 2; edges 0–1–2; final optimized QCO with terminal readout    |

The intermediate elision artifacts were captured with
`--mlir-disable-threading --mlir-print-ir-after-all --mlir-print-ir-module-scope`.
For `measured.mlir`, append `bit[9] c = 0;`, followed by `c[7] = measure q[0];`,
`c[2] = measure q[1];`, and `c[5] = measure q[2];` to the source circuit.

For the sparse target, `initial = [1,0,2]`, `routing = [1,2,0]`, and
`sites = [10,30,20]`: source inputs occupy sites `[30,10,20]`, and source
outputs occupy sites `[20,30,10]`. This composition is deliberately asymmetric.
Independent NumPy gate expansion and row/column permutation, without Core DDs or
Core output normalization, gave maximum matrix-entry residual
`2.531698018113677e-15` for both the sparse-site and workspace fixtures.
Mutation checks detected wrong angles, controlled operations, phase, layouts,
site metadata, and sparse destinations; reducing `input_count` from 3 to 2 was
rejected before comparing matrices.
