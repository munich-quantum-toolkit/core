# DDSIM replacement and measured simplification

Status: implemented and locally validated; published Core wheels and release-
candidate CI remain release gates.

## Scope and ownership

DDSIM is a pure Python compatibility package over Core. Keep Qiskit and OpenQASM
sampling, pure-state and unitary simulation, explicit QIR execution, and V2
primitives. Core owns import, lowering, DD execution, and exact Pauli
estimation. DDSIM owns retained names, backend results, and CLI behavior.

An exact estimate uses an uncollapsed vector DD, not a dense amplitude array.
Reset and measurement-dependent state preparation remain sampled-estimator
workloads. Base QIR contains unitary preparation and terminal measurements;
reset is outside that profile. Arbitrary executable QIR has no unitary-export
contract. No HSF, noise engine, or alternative simulation scheme is ported.

## Decisions and evidence

Reuse Qiskit PUB containers, binding, broadcasting and sampled primitives. Keep
validation at ownership, numerical and IR boundaries. Map conversion owns
observable strings while the binding releases the GIL; collection is already
threshold based. Package reuse and alternative binding adapters add semantics
without a demonstrated benefit and are not included.

Construct each Pauli string with the existing tensor-product operation. The
ascending loop adds an uncontrolled factor above all existing wires; disabling
index shifting preserves identity gaps and physical wire positions. Independent
dense oracles cover all three-qubit strings; a separate helper probe checked all
340 strings of widths one through four with exact matrix agreement.

On the existing 48-qubit GHZ public estimator workload, the substitution reduced
tracked allocation volume by 490800 bytes (11.6%) and peak tracked live
allocation by 15.3%. Seven alternating pairs of 100 warmed calls measured median
wall time 1.90 ms before and 1.84 ms after. This is a modest workload-specific
result on a shared host, not a general speedup claim. Process peak RSS is not
allocation volume and did not establish a corresponding reduction. Independent
dense results and analytical compact-state results remained unchanged.

DDSIM uses `uv_build` and the pure Python MQT repository setup. Equivalent
Hatchling and uv_build artifacts contained identical package payloads; the
bounded warm build comparison favored uv_build. Remove redundant nox syncing and
stale warning filters. Use a Qiskit Git overlay in the existing nightly session
so synchronization does not undo the selected dependency.

## Validation and release limits

Run `ctest --preset release`, the DD Debug tests, Python tests,
`uvx nox -s cpp-lint`, `uvx nox -s lint`, stub generation and executable
documentation. Installed-wheel checks exercise runtime discovery, QIR
text/bitcode, sampling, state and unitary results, primitives and CLI outside
the source checkout. Existing ordering, phase and binding tests pass normally
and fail with the corresponding injected defects. One independent Ponytail
review found no new actionable defect. Keep raw profiles, emitted IR and
reproducible commands in the external evidence bundle, not as a new repository
benchmark framework.

DDSIM's development dependency pins an exact Core Git revision. Before release,
replace it with the first published Core version containing the required APIs,
then run installed-wheel checks and release-candidate CI across the declared
existing matrix. Source installation of Core requires its native build
toolchain.

The separately reviewable Core changes are listed in
[the Core PR ledger](ddsim-core-pr-candidates.md).
