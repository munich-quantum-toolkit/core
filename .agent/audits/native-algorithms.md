# SpecAudit: native C++ Algorithms

Scope:

- production: the eight public headers under `include/mqt-core/algorithms/` and
  their eight matching translation units under `src/algorithms/`;
- focused tests: all ten files under `test/algorithms/`;
- focused target: `mqt-core-algorithms-test`;
- baseline: `cb5cf0103bd9841726c8ec6c5abb725758afea58`;
- audit date: 2026-08-20.

Every repository citation below was read at the pinned baseline. All audit,
prosecution, defence, and execution worktrees were clean before their role ran.
Every mutation ran in a disposable detached worktree. The executor restored the
baseline after each probe and finished with all 280 Algorithms tests passing. No
production or test change survives. This file is the audit ledger only. It does
not apply a verdict.

## Role registry

The audit used these isolated roles:

- persistent scope steward;
- machine-checked, published, requested, and public-surface cartography;
- assertion census;
- three independent prosecutors, split by algorithm family;
- two provenance tracers;
- two independent defenders;
- a serialized executor and a supplemental executor;
- unlock and architecture analysis;
- fresh red team;
- fresh final adjudicator.

Role isolation used detached worktrees and fresh agent contexts. Cartographers
did not read tests. Prosecutors did not read provenance or other prosecution.
Defenders did not read prosecution. The executors received the frozen ledger,
challenged assertion IDs, and prescribed probes. The final adjudicator first
reconstructed the census and source evidence, then challenged the proposed
one-to-one mapping against the executed evidence.

## Spec ledger

**S1, rungs 1 and 2.** MQT Core exports `MQT::CoreAlgorithms` as a public
library. Its public file set contains the Algorithms headers
(`src/algorithms/CMakeLists.txt:9-29`). The public surface is the declarations
in the eight headers listed in this audit. A declaration promises that the
factory or helper exists. It promises a result only where a later ledger entry
states the result.

**S2, rungs 1 to 3.** Bernstein--Vazirani recovers a supplied hidden parity
string. The public surface includes ordinary and iterative factories for an
explicit `BVBitString`, a qubit count and seed, or both an explicit string and
width. `BVBitString` has 4096 bits
(`include/mqt-core/algorithms/BernsteinVazirani.hpp:26-43`).
[PR #22](https://github.com/munich-quantum-toolkit/core/pull/22) requested the
corrected implementation, dynamic form, arbitrary-width generation, and tests.

**S3, rungs 1 to 3.** `createGHZState` constructs a GHZ-state circuit
(`include/mqt-core/algorithms/GHZState.hpp:11-22`). The promised state is
`(|0...0> + |1...1>) / sqrt(2)`, up to global phase. The maintainer request in
[PR #445](https://github.com/munich-quantum-toolkit/core/pull/445)
also requires both endpoint amplitudes to be `1/sqrt(2)` and requires equality
between circuit simulation and `makeGHZState`.

**S4, rungs 1 to 3.** Grover's public surface includes initialization, oracle,
diffusion, iteration-count, explicit-target, and seeded factories
(`include/mqt-core/algorithms/Grover.hpp:25-36`). The semantic promise is
amplification of one supplied target with a suitable iteration count near
`pi*sqrt(N)/4`. `PR #22` requested arbitrary-qubit generation and tests.

**S5, rungs 1 to 3.** The QFT surface contains the ordinary and iterative
factories (`include/mqt-core/algorithms/QFT.hpp:11-25`). The ordinary factory
implements the normalized `2^L`-point quantum Fourier transform. The iterative
factory is the measured semiclassical replacement immediately before
measurement. `PR #22` requested the dynamic form. The mathematical source is
[Griffiths and Niu](https://arxiv.org/abs/quant-ph/9511007).

**S6, rungs 1 and 2.** QPE estimates the eigenphase of a unitary eigenvalue
`exp(2*pi*i*phi)`. The public surface contains ordinary and iterative factories
for a generated instance or an explicit `lambda` and precision
(`include/mqt-core/algorithms/QPE.hpp:11-34`). `PR #8` and its cited paper
define the requested ordinary, iterative, simulation, and transformation
workflows:
[PR #8](https://github.com/munich-quantum-toolkit/core/pull/8),
[Handling non-unitary quantum operations](https://arxiv.org/abs/2106.01099).

**S7, rung 3, reported defect.** `PR #22` states that QPE must construct above
the historical 32-qubit failure boundary. It does not promise an exact upper
bound or register layout.

**S8, rung 3, reported defect.**
[PR #417](https://github.com/munich-quantum-toolkit/core/pull/417) states that
random inexact QPE construction must be reliable. It does not promise a gate
sequence, layout, numeric instance, bit order, or distribution.

**S9, rungs 1 and 2.** `createRandomCliffordCircuit` constructs a Clifford
circuit from a qubit count, depth, and seed
(`include/mqt-core/algorithms/RandomCliffordCircuit.hpp:11-26`). No source
promises one random distribution, operation sequence, or exact realized depth.

**S10, rungs 1 and 2.** `createStatePreparationCircuit` prepares the supplied
normalized complex state
(`include/mqt-core/algorithms/StatePreparation.hpp:23-46`). The changelog
publishes the feature (`CHANGELOG.md:689-700`). `PR #543` records a defect in
the global phase of the simulated prepared state, so component-wise complex
output checks are regression anchors rather than excess phase constraints.

**S11, rung 1.** State preparation requires a normalized vector whose length is
a power of two. It throws `invalid_argument` when either condition is false
(`include/mqt-core/algorithms/StatePreparation.hpp:37-45`).

**S12, rungs 1 to 3.** `createWState` constructs a W-state circuit
(`include/mqt-core/algorithms/WState.hpp:11-22`). The promised state is the
equal superposition of all one-excitation basis states, up to global phase.
`PR #445` requested an exact cross-check between circuit simulation and
`makeWState`.

**S13, rung 2.** The current public Algorithms API uses factory functions. The
factory shape is published. No source promises identity with a removed class
hierarchy, internal decomposition, or permanence through MQT Core v4.

**S14, rung 3.** An explicit seed initializes a local generator for each BV,
Grover, QPE, and random-Clifford call. Other factory calls cannot perturb the
result through hidden `QuantumComputation` RNG state (`UPGRADING.md:175-179`).
`Issue #2100` and `PR #2111` require reproducibility and seed-separation
coverage:
[issue #2100](https://github.com/munich-quantum-toolkit/core/issues/2100),
[PR #2111](https://github.com/munich-quantum-toolkit/core/pull/2111).

## Explicit non-promises

No rung 1 to 3 source promises:

- an algorithm circuit name or a result encoded in that name;
- one qubit or classical-bit order, output permutation, or ancilla layout;
- one gate decomposition, operation count, DD shape, table size, or root edge;
- one observable global phase, except for the reported state-preparation defect;
- exact equality of static and dynamic garbage or pre-measurement states;
- a universal Grover success floor or one iteration-rounding rule;
- a QFT swap placement, operation order, or zero-width behavior in the frozen
  ledger;
- QPE `lambda` units, generated distribution, bit order, precision range, or
  dynamic register shape;
- injective seeds, one random engine or distribution, or cross-version random
  output;
- an exact random-Clifford circuit for a seed;
- one W-state sampling frequency or a guarantee that finite samples contain
  every supported outcome.

## GitHub drift and ecosystem overlap

The audit refreshed open issues, open pull requests, and downstream consumers on
2026-08-20. The repository had 47 open issues and 27 open pull requests.

- Open issue [#2095](https://github.com/munich-quantum-toolkit/core/issues/2095)
  proposes reducing or removing `MQT::CoreAlgorithms`. Its acceptance criteria
  require a consumer census before deletion. This audit therefore treats target
  removal as a migration question, not a test-cleanup consequence.
- No open pull request touched an audited Algorithms header, implementation, or
  test file at the snapshot.
- Open issue [#1115](https://github.com/munich-quantum-toolkit/core/issues/1115)
  and open PR [#2135](https://github.com/munich-quantum-toolkit/core/pull/2135)
  overlap only through MLIR QFT and QPE examples or benchmarks.
- QCEC, DDSIM, QUSAT, and `ystade/eval-qir-backend` consume Algorithms
  factories. QCEC also parses the Bernstein--Vazirani circuit-name suffix.
  Name-independent Core tests therefore do not yet unlock removal of the
  production name encoding. QCEC must migrate first.

## Human decision

On 2026-08-20, the maintainer accepted the recommended resolution slate. The
maintainer confirmed that QCEC permits observable-result equivalence for the
dynamic algorithm tests. The resolution must preserve all 35 anchors, add the
identified replacement coverage before removing accidental substitutes, and keep
target removal and the CircuitOptimizer assertions deferred. The audit and
accepted remedies will share one pull request with separate commits.

## Assertion census

The audited target contains exactly 88 lexical GoogleTest assertion sites:

| File                             | Sites |
| -------------------------------- | ----: |
| `eval_dynamic_circuits.cpp`      |     4 |
| `test_bernsteinvazirani.cpp`     |     5 |
| `test_entanglement.cpp`          |     4 |
| `test_grover.cpp`                |     9 |
| `test_qft.cpp`                   |    35 |
| `test_qpe.cpp`                   |    14 |
| `test_random_clifford.cpp`       |     2 |
| `test_randomized_algorithms.cpp` |     8 |
| `test_statepreparation.cpp`      |     6 |
| `test_wstate.cpp`                |     1 |

Stable IDs combine the family prefix with the pinned-baseline source line. The
complete one-to-one ID mapping appears in the summary.

`G-38` is one lexical fixture assertion. It runs after three test templates but
is not three census entries. The ten test files expand to 280 discovered test
cases. Their assertion loops execute 376,552 times on the passing path, chiefly
because the QFT tests enumerate exponentially many matrix entries in duplicate
construction paths.

The scope scan also found 20 direct-use assertions in
`test/circuit_optimizer/test_flatten_operations.cpp`. They belong to
`mqt-core-circuit-optimizer-test` and test flattening, not Algorithms behavior.
This audit records and defers those assertions to the CircuitOptimizer owner.
They are not part of the 88-site arithmetic.

## Summary

The final adjudication classifies every one of the 88 assertion sites:

- 35 anchored;
- 31 redundant;
- 20 over-specified;
- 2 contract-free;
- 0 coverage-driven.

Thus `35 + 31 + 20 + 2 = 88`.

The exact one-to-one mapping is:

- Anchored: `EDC-188`, `EDC-361`; `GHZ-56`, `GHZ-57`, `GHZ-64`; `G-38`, `G-121`,
  `G-175`; `QFT-112`, `QFT-122`, `QFT-164`, `QFT-190`, `QFT-214`, `QFT-220`,
  `QFT-245`; `QPE-155`, `QPE-167`, `QPE-168`, `QPE-199`, `QPE-200`, `QPE-205`,
  `QPE-211`, `QPE-212`; `RNG-24`, `RNG-26`, `RNG-28`, `RNG-29`, `RNG-30`,
  `RNG-32`, `RNG-44`; `SP-82`, `SP-83`, `SP-91`, `SP-101`; and `W-57`.
- Redundant: `QFT-85`, `QFT-88`, `QFT-99`, `QFT-106`, `QFT-113`, `QFT-115`,
  `QFT-116`, `QFT-127`, `QFT-130`, `QFT-133`, `QFT-141`, `QFT-148`, `QFT-154`,
  `QFT-155`, `QFT-157`, `QFT-158`, `QFT-169`, `QFT-172`, `QFT-178`, `QFT-184`,
  `QFT-185`, `QFT-194`, `QFT-205`, `QFT-208`, `QFT-212`; `QPE-141`, `QPE-144`;
  `RC-54`, `RC-64`; `SP-75`; and `SP-78`.
- Over-specified: `EDC-478`, `EDC-591`; `BV-66`, `BV-84`, `BV-99`, `BV-114`,
  `BV-149`; `G-118`, `G-119`, `G-172`, `G-173`, `G-183`, `G-188`; `QFT-91`,
  `QFT-200`; `QPE-140`, `QPE-175`, `QPE-247`, `QPE-277`; and `RNG-45`.
- Contract-free: `GHZ-54` and `QFT-249`.

The mapping counts a loop assertion once. It also counts `G-38` once even though
the fixture runs it after three test templates.

The highest-value work is not wholesale deletion. It is to replace self-derived
name oracles, collapse duplicate QFT matrix traversals, remove no-throw
wrappers, narrow whole-state dynamic comparisons to observable semantics, and
move DD resource checks to the package that owns the resource.

## Verdicts and remedies

### 1. Name-derived result oracles couple behavior to metadata

The four sampled BV tests (`BV-66`, `BV-84`, `BV-99`, and `BV-114`) derive the
expected hidden string from `QuantumComputation::getName()`. `EDC-478` also
derives its hidden value from the generated name. The Grover fixture derives its
target from the generated name before `G-118`, `G-119`, `G-121`, `G-172`,
`G-173`, `G-175`, `G-183`, and `G-188` run.

The name is not in S2 or S4. A metadata-only prefix change failed the old tests
while leaving circuit behavior unchanged. A replacement that supplies an
explicit hidden string or target passed all 70 focused cases and all 280 target
cases. Removing one BV parity gate failed 22 of 23 affected explicit-result
cases; removing Grover diffusion failed all 25 affected cases.

Remedy: construct ordinary and iterative BV from the same explicit-width input.
Use the input as the oracle. Keep the large-width seeded factories as
construction and seed regressions without parsing their names. Construct Grover
from an explicit target and use that target for amplitude and sampling checks.
Do not remove production name encoding until QCEC migrates.

### 2. Whole-state dynamic comparisons exceed observable semantics

`BV-149`, `EDC-478`, `EDC-591`, `QPE-247`, and `QPE-277` compare a complete
state or functionality after dynamic-circuit rewrites. The static and dynamic
algorithms promise the same result, not the same garbage, layout, global phase,
or coherent state immediately before measurement.

A BV garbage-only change preserved every classical recovery result and failed
only the zero-input fidelity case. A `Z` immediately before each iterative QFT
measurement preserved the measured distribution but failed `EDC-591`. The same
kind of diagonal pre-measurement change preserved iterative QPE samples but
failed the full functionality equality and two inexact fidelity instances. An
iterative-only global phase passed the QPE dynamic checks, which confirms that
the existing pair is inconsistent about which excess state details it pins.

Remedy: compare decoded, observable distributions or deterministic classical
results. Normalize only conventions that are explicitly supported. Retain small
independent semantic tests for the ordinary and iterative factory, and retain
separate transformation-owner tests for reset elimination and measurement
deferral.

### 3. QFT tests mix semantics with exact DD representation

The QFT matrix, recursive-matrix, simulation, and sampling groups contain 35
assertions. They repeat construction, exact DD size, real-table count, root
weight, first-row and first-column values, cleanup, and result checks across
four paths. The full passing run executes 374,652 QFT assertions.

Exact node counts (`QFT-91`, `QFT-133`), exact real-table counts (`QFT-99`,
`QFT-141`), and root weights (`QFT-106`, `QFT-148`, `QFT-184`, `QFT-185`) expose
the current canonical DD representation. They are not QFT surface promises. The
first row and column are mathematical QFT values, but duplicating both
components through sequential and recursive construction adds little fault
separation. Appending or prepending one phase gate made each imaginary check
fail together with its corresponding real check. A deleted controlled phase or
output reversal changed the exact DD-size assertions while the current first
row, first column, and `|0>` sampling oracles did not notice the semantic fault.

Remedy: retain one bounded, semantic full-matrix oracle at small widths. It must
check nontrivial controlled phases and output order. Retain one measured
iterative-distribution check. Test sequential and recursive DD construction in
the DD owner. Do not use Algorithms to freeze node count, unique-table count,
root normalization, or construction strategy. This replacement removes most of
the exponential duplicate work while improving mutation sensitivity.

### 4. Some cleanup checks protect resources; others inspect the wrong owner

`QFT-122`, `QFT-164`, and `QFT-220` detect unreleased matrix roots or recursive
intermediates after explicit owner release. `G-38` detects the historical Grover
real-table lifetime and immortalization defects on the functionality paths.
`PR #1020` records both classes of defect. These checks are safety anchors.

The same lexical `G-38` runs after Grover simulation even though `dd::sample`
owns a separate internal package. `QFT-249` also inspects an untouched fixture
package after `dd::sample`. Neither instance can observe a sampler leak.
`QFT-200` compares only the real-number count and did not fail when the
simulated vector root was left referenced.

Remedy: retain the matrix and Grover functionality lifetime checks. Do not count
the simulation instance of `G-38` as distinct coverage. Remove `QFT-249`.
Replace `QFT-200` with an owner-level DD test that accepts the package used by
simulation and checks active nodes or roots, not only the real-number count.

### 5. Explicit no-throw wrappers add no oracle

The 15 wrappers `SP-75`, `SP-78`, `QFT-85`, `QFT-88`, `QFT-127`, `QFT-130`,
`QFT-169`, `QFT-172`, `QFT-205`, `QFT-208`, `QFT-212`, `QPE-141`, `QPE-144`,
`RC-54`, and `RC-64` are redundant. Replacing each wrapper with the same bare
call preserved all 80 focused cases and all 280 target cases. Representative
factory, DD-builder, and simulator throws still failed the tests as uncaught
exceptions.

Remedy: keep every call and remove only `ASSERT_NO_THROW`. Keep all 16 random
Clifford simulation repetitions. The audit found no evidence that reducing those
repetitions preserves the randomized stress coverage.

### 6. Grover component checks pin phase beyond the probability contract

`G-118`, `G-119`, `G-172`, and `G-173` pin the real and imaginary components of
the target amplitude. S4 promises amplification, not a global phase. Removing
the four component checks while keeping `G-121` and `G-175` passed all 50
focused cases and all 280 target cases. Removing diffusion then failed all 50
focused cases through the retained probability checks. The component checks are
over-specified, not merely duplicate, because a legal circuit-wide phase makes
them fail while the probability checks pass.

Remedy: keep a phase-insensitive target-probability check for sequential and
recursive construction. Use an explicit target. Remove the component checks. For
sampling, merge `G-183` into a safe lookup used by `G-188`; presence is not a
second semantic requirement.

### 7. Fixed different-seed inequality is stronger than seed separation

`RNG-45` requires seeds 23 and 24 to yield different BV circuits. S14 requires
local deterministic RNG state and meaningful seed sensitivity. It does not make
the seed-to-circuit map injective. A collision-tolerant replacement that
requires at least two distinct results across a small seed set passed. A fault
that ignored the seed failed that replacement.

The six same-call equality assertions and `RNG-44` are anchors. A per-call nonce
fault failed all six same-seed assertions. A hidden cross-factory epoch fault
left immediate same-factory equality intact and failed only `RNG-44`.

Remedy: retain the seven anchored assertions. Replace `RNG-45` with a bounded
diversity check across several seeds. Do not publish a collision-free seed pair
or random-engine sequence.

### 8. GHZ operation count is not a state contract

`GHZ-54` requires exactly `nq` operations. S3 does not promise a decomposition
or topology. A global `-1` phase implemented with extra gates preserved the GHZ
state and invalidated the exact count. The exact endpoint checks and the direct
DD equality are explicit `PR #445` requirements. `GHZ-56`, `GHZ-57`, and
`GHZ-64` therefore remain anchors.

Remedy: remove the exact operation count. If large-width construction needs a
resource guard, add a separately stated asymptotic bound that permits valid
linear decompositions.

### 9. QPE shape checks pin one implementation

`QPE-140` requires `precision + 1` qubits and `QPE-175` requires exactly two
qubits. An idle garbage ancilla preserved every ordinary and iterative result
oracle after the test package used the circuit's actual width. Only the two
shape checks failed.

Remedy: remove exact register-size checks. Keep the mathematical result and
probability checks. If resource bounds become public, test a bound rather than
one layout.

### 10. Reported defects and contract checks must remain

The exact and inexact dynamic-QPE construction paths `EDC-188` and `EDC-361` are
regression anchors, subject to replacing inferred `lambda` metadata with an
explicit input. Historical 32-bit shifts fail the exact setup at precisions 36,
41, and 61. The current inexact test does not detect `PR #417`'s loop defect, so
S8 needs a new direct generated-inexact regression.

`QPE-155`, `QPE-167`, `QPE-168`, `QPE-199`, `QPE-200`, `QPE-205`, `QPE-211`, and
`QPE-212` kill historical phase, inverse-QFT, feedback, or endianness faults.
`SP-82` and `SP-83` kill `PR #543`'s state-preparation global-phase defect;
fidelity alone did not. `SP-91` and `SP-101` each failed when its matching
validation was bypassed. `W-57` protects support across all one-excitation
outcomes. The seeded RNG checks protect S14.

Remedy: preserve these semantic and defect anchors. Narrow their setup only
where it depends on unspecified metadata or layout.

## Executed evidence

The executor configured and built the focused release target, then ran:

```console
cmake --preset release
cmake --build --preset release --target mqt-core-algorithms-test
./build/release/test/algorithms/mqt-core-algorithms-test
```

The baseline result was 280 of 280 passing. Each probe used the matching
GoogleTest filter. The executor restored and verified the pinned baseline after
every probe. The principal probe families were:

- T1 deletion or narrowing of one challenged assertion, followed by the focused
  and full target tests;
- T2 a semantics-preserving alternative that should fail only an excess oracle;
- T2 a real fault that the replacement must kill;
- T3 survivor comparison across small fault families where one mutation could
  not settle redundancy.

Executed results:

- removing all 15 no-throw wrappers preserved focused and full results;
- name-independent BV and Grover remedies passed; metadata-only renames passed;
  parity and diffusion faults failed the replacements;
- a GHZ global phase passed a fidelity probe, while a missing entangler failed;
  exact `PR #445` checks were retained despite that narrower mathematical
  result;
- phase-insensitive Grover probability checks killed a missing diffusion;
- a collision-tolerant seed check killed a seed-ignore fault;
- the QFT matrix fault family included `H` to `X`, deleted and negated
  controlled phases, missing output reversal, root phase, sequential-only,
  recursive-only, iterative-only, and resource-owner faults;
- QPE probes included idle ancillas, historical inverse-QFT and feedback faults,
  global phase, pre-measurement diagonal phase, historical 32-bit shifts, and
  the historical random-inexact loop condition;
- bypassing each state-preparation validation failed its exact exception test;
- removing the state-preparation global-phase correction failed component checks
  but passed a fidelity-only replacement;
- an exact W-state circuit-versus-DD comparison killed a relative-phase fault
  that the current support-only sampling assertion missed.

No source-coverage run was supplied or executed. Each wrapper remedy preserves
its production call, and each semantic remedy must land with its replacement
oracle. The audit therefore records no coverage-driven assertion.

The final disposable-worktree check reported the pinned HEAD, empty index and
worktree diffs, no untracked files, and 280 of 280 baseline tests passing.

## Provenance

[Commit `64b590b0`](https://github.com/munich-quantum-toolkit/core/commit/64b590b0)
from `PR #8` co-introduced QPE, iterative QPE, the dynamic transformations, and
the original QPE assertions. The assertions did not first exist as a failing
test revision. Later test-only changes altered output indexing and shot count.

[Commit `8f13b506`](https://github.com/munich-quantum-toolkit/core/commit/8f13b506)
from `PR #22` co-introduced or rewrote the dynamic BV and QFT tests, the sampled
BV oracles, and the above-32 QPE fix. The exact-QPE dynamic range reaches 61 and
therefore covers that defect indirectly. The inexact generated-factory defect
fixed by `PR #417` received no focused assertion.

The original GHZ operation and endpoint assertions arrived with the GHZ
implementation. `PR #445` later added exact circuit-simulation versus direct-DD
equality after a maintainer requested that expression. The same request added an
exact W-state cross-check, but a later state-generation refactor removed the W
cross-check and left only sampled support coverage.

The Grover amplitude and probability checks arrived with Grover production.
Their paths and tolerances changed with later DD migrations. The cleanup check
predates mark-and-sweep, was disabled for a numerical garbage-collection defect,
and was restored and changed again with DD fixes. `PR #1020` introduced its
current immortal-value form while fixing real lifetime and collection defects.

The QFT matrix-shape assertions arrived with the implementation and changed with
DD representation migrations. Exact table counts, root access, and cleanup
counts followed DD API and garbage-collection changes. This co-evolution is
strong provenance evidence that the representation checks belong to the DD
owner, even where they happen to kill an Algorithms mutation.

`PR #543` introduced state-preparation output-component checks and then fixed a
simulation-global-phase defect that those checks exposed. Replacing them with
fidelity would erase that regression. `PR #2111` introduced the RNG-locality
tests and `RNG-45` together with the local-generator implementation. That commit
records Codex assistance. No other relevant introducing or semantic-change
commit found by the provenance roles records AI-assistance trailers.

## Coverage gaps

1. S8 lacks a direct generated-inexact QPE regression. Add a deterministic
   construction check that confirms the generated phase is genuinely inexact
   without freezing its exact value or circuit.
2. S12 has support coverage but no equal-amplitude or relative-phase oracle.
   Restore the `PR #445` circuit-versus-`makeWState` cross-check, or use an
   equivalent state-semantic oracle.
3. The retained QFT row, column, and `|0>` checks do not detect controlled-phase
   or output-order faults. Replace duplicate exponential slices with one small
   full-matrix semantic oracle.
4. Sampler resource safety has no owner-visible assertion. Add a DD-level API or
   fixture that can inspect the package used by sampling.
5. The exact-QPE above-32 regression is embedded in a broad transformed-unitary
   test. Add a narrow construction regression above 32.

## Unlock and architecture analysis

The audit does not justify removing `MQT::CoreAlgorithms` today. `Issue #2095`
already owns that migration, and active consumers still exist. Test cleanup does
unlock smaller steps:

- explicit input oracles decouple tests from circuit-name metadata;
- a small semantic QFT oracle can replace hundreds of thousands of duplicate
  assertion executions;
- owner-level DD cleanup checks permit Algorithms tests to stop depending on
  unique-table representation;
- observable dynamic comparisons permit alternate reset, measurement-deferral,
  ancilla, and layout implementations;
- collision-tolerant RNG coverage permits engine and distribution changes;
- removing no-throw wrappers and duplicate phase-component checks reduces noise
  without reducing fault sensitivity.

Production name removal remains blocked by QCEC's direct parser. Target removal
remains blocked by the wider consumer migration required by `issue #2095`. No
production deletion should be bundled with a test-resolution pull request.

## Red-team revisions and residual risks

The red team rejected four tempting but unsafe conclusions:

- state-preparation component checks are reported-defect anchors, not removable
  global-phase over-specification;
- GHZ exact amplitude and DD equality checks have an exact maintainer request;
- random-Clifford's 16 repetitions must remain until sanitizer or mutation
  evidence supports a smaller stress sample;
- fixed-seed inequality is over-specified, but its seed-sensitivity purpose must
  survive in a collision-tolerant form.

QFT is the main residual judgment risk. Some exact DD representation checks
currently kill QFT faults only because the fault changes the DD shape. Deleting
them without first adding the small semantic full-matrix oracle would reduce
defect sensitivity. Resolution must land the replacement before or in the same
commit that removes those checks.

## Deliberately not touched

- no production or test source changed;
- no assertion under `test/circuit_optimizer/` was adjudicated;
- no public algorithm or target was removed;
- no downstream repository was changed;
- no issue, pull request, comment, review, or branch was changed remotely;
- no generated file, dependency, or benchmark was changed.

## Progress

- [x] Pin a clean baseline and freeze the scope.
- [x] Build a source-ranked ledger.
- [x] Refresh open issue, pull-request, and consumer overlap.
- [x] Census all local and transitive assertions.
- [x] Run isolated prosecution, provenance, and defence.
- [x] Execute narrowing, alternative, fault, and resource probes.
- [x] Run unlock, architecture, and red-team reviews.
- [x] Insert the final one-to-one adjudication table and arithmetic.
- [x] Run the repository lint suite on this audit ledger.
