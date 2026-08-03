# Superconducting QDMI device

The MQT Core superconducting (SC) provider builds a session-owned runtime model
from strict JSON. Separate sessions using the same provider library and prefix
can expose different names, capacities, topology, operations, and calibration.

## Configuration sources

The provider selects the first available source in this order:

1. QDMI CUSTOM1 containing NUL-terminated inline JSON;
2. QDMI CUSTOM2 containing a NUL-terminated JSON file path;
3. `MQT_CORE_QDMI_SC_CONFIG_JSON`;
4. `MQT_CORE_QDMI_SC_CONFIG_FILE`;
5. `mqt-core-qdmi-sc-device.json` beside the provider shared library.

When direct QDMI callers populate both explicit slots, inline JSON wins. Setting
both SC environment sources is an error. Configuration cannot be changed after
successful initialization. A failed initialization does not commit partial
state, so the same allocated session can be corrected and initialized again.

## Schema and calibration

Every description requires `"schema-version": 1`, a name, positive `numQubits`,
a duration unit, qubit properties, ordered couplings, and operations. Unknown
and missing fields are rejected. Each operation declares its name, parameter
count, arity, optional supported ordered site tuples, optional default duration
and fidelity, and optional tuple-specific overrides.

```json
{
  "schema-version": 1,
  "name": "Small SC device",
  "numQubits": 2,
  "durationUnit": {"unit": "ns", "scaleFactor": 1.0},
  "qubitProperties": {
    "defaults": {"t1": 100000, "t2": 150000},
    "overrides": [{"qubit": 1, "name": "QB2", "t1": 95000}]
  },
  "couplings": [[0, 1]],
  "operations": [
    {
      "name": "cz",
      "numParameters": 0,
      "numQubits": 2,
      "duration": 200,
      "fidelity": 0.99,
      "siteOverrides": [
        {"sites": [0, 1], "duration": 180, "fidelity": 0.995}
      ]
    }
  ]
}
```

`durationUnit.unit` is one of `s`, `ms`, `us`, and `ns`, and its `scaleFactor`
is positive and finite. Couplings contain distinct, valid, non-self tuples.
Their order is significant; list both orientations if an operation supports
both. Operation names are unique, durations are non-negative, and fidelities are
finite values in the inclusive range `[0, 1]`.

The `sites` member may be omitted for one-qubit operations to support every
qubit, or for two-qubit operations to use the coupling map. Higher-arity
operations require explicit tuples. Every explicit two-qubit tuple must be an
exact ordered edge in the coupling map. Every site override must select one of
the operation's supported tuples and supply a duration, fidelity, or both.

When operation duration or fidelity is queried, a matching tuple override wins
over the operation default. Missing data returns `QDMI_ERROR_NOTSUPPORTED`. Each
qubit override may also provide a non-empty site name. Qubit T1 and T2 use a
per-qubit override first and the qubit default second. The order of a configured
site tuple is significant. Handles from another session are rejected.

The calibration values bundled in `json/sc/mqt-core-qdmi-sc-device.json` are
synthetic examples for testing and documentation; they are not measurements of
physical hardware.

## Bundled IQM models

The SC provider also installs `json/sc/iqm-garnet.json` and
`json/sc/iqm-emerald.json`. They model the 20-qubit, 30-edge Garnet and
54-qubit, 90-edge Emerald architectures and expose the portable IQM operation
set `r`, `cz`, and `measure`. Their stable registry IDs are `mqt.sc.iqm.garnet`
and `mqt.sc.iqm.emerald`.

The snapshots were derived from authenticated IQM architecture and calibration
data retrieved on 2 August 2026 through
[QDMI-on-IQM](https://github.com/iqm-finland/QDMI-on-IQM). Site names and
topology came from each static architecture, operation loci from its default
dynamic architecture, and available T1, T2, and per-locus fidelities from its
default calibration-set quality metrics. Coherence times were rounded to the
nearest nanosecond and stored as integral values using unit `us` and scale
factor `0.001`. They are historical compiler-facing snapshots, not claims about
current hardware calibration.

The IQM interface did not report operation durations, so the files intentionally
omit them.
