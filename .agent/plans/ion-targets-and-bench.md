# Current native targets in Core and Bench

Status: superseded by the current
[native gate decision record](native-ion-gate-targets.md) and
[Bench target documentation](https://github.com/munich-quantum-toolkit/bench/blob/feat/mqt-core-compiler/docs/targets.md).

The [joint audit](../audits/ion-targets-and-bench.md) retains the hardware
evidence and findings. Core owns gate semantics and Qiskit capability import;
Bench owns its catalogue, native-versus-mapped placement policy, and output
contract. Both use radians and preserve exact global phase. Arbitrary virtual RZ
and standard fixed RX aliases are supported.
