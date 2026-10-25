# Private workspace checker eligibility

Inputs are the source fixtures
`mlir/unittests/Dialect/QCO/Utils/testdata/verification/before-elision.mlir` and
`workspace.mlir`. The latter has a private fourth wire. The mutation inserts an
X on that wire immediately before the return:

```mlir
%workspace = qco.static 3 : !qco.qubit
%dirty = qco.x %workspace : !qco.qubit
qco.sink %dirty : !qco.qubit
```

The pre-correction default Python `mlir.verify` call returned non-equivalence
from simulation and cancelled construction before its required clean-output
proof. The following is a transcription of observed tool stdout from the root
investigation; no raw output file was retained:

```text
VerificationVerdict.NOT_EQUIVALENT sample exceeds the all-input entry-residual bound
[(VerificationMethod.SIMULATION, VerificationVerdict.NOT_EQUIVALENT, 'sample exceeds the all-input entry-residual bound'), (VerificationMethod.CONSTRUCTION, VerificationVerdict.INCONCLUSIVE, 'DD computation cancelled.')]
```

Equivalent reproduction code, from the repository root with the native Python
extension installed, is:

```python
from pathlib import Path
from mqt.core import mlir

fixtures = Path("mlir/unittests/Dialect/QCO/Utils/testdata/verification")
source = mlir.QCOProgram.from_mlir_str((fixtures / "before-elision.mlir").read_text())
dirty_text = (
    (fixtures / "workspace.mlir")
    .read_text()
    .replace(
        "    return",
        """
    %workspace = qco.static 3 : !qco.qubit
    %dirty = qco.x %workspace : !qco.qubit
    qco.sink %dirty : !qco.qubit
    return""",
    )
)
dirty = mlir.QCOProgram.from_mlir_str(dirty_text)
result = mlir.verify(source, dirty)
print(result.verdict, result.reason)
```

Finite samples cannot establish the required clean-workspace premise over all
quantum inputs. Automatic checking now uses construction for this domain, and
forced simulation is unsupported. Initialized checking with no free quantum
inputs remains eligible for simulation because it can check the complete
supplied state. Alternating checking already requires every wire to be a free
input, and terminal-effect checking already requires construction.

The updated `VerificationInterfaceTest.PrivateWorkspaceMustReturnToZero`
exercises these expectations:

| Input and checker                           | Expected    | Observed    |
| ------------------------------------------- | ----------- | ----------- |
| Original clean workspace, construction      | Equivalent  | Equivalent  |
| Mutated dirty workspace, construction       | Unsupported | Unsupported |
| Mutated dirty workspace, automatic          | Unsupported | Unsupported |
| Original clean workspace, forced simulation | Unsupported | Unsupported |

All assertions passed in the full 300-test QCO Utils run recorded at
`/tmp/native-equivalence-final-workspace-eligibility-qco-utils.log`. The
retained summary is in [native-test-results.txt](native-test-results.txt).

The final rebuilt Python API was also checked with the reproduction above. Its
[captured stdout](workspace-after.txt) reports `UNSUPPORTED` with only the
construction checker selected.
