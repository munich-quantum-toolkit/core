---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# IQM QDMI Device

MQT Core can integrate [IQM's QDMI device](https://github.com/iqm-finland/QDMI-on-IQM/) through the external {code}`iqm-qdmi` package and expose it through the same Qiskit-facing QDMI backend surface as the in-tree devices.
Unlike the DDSIM and neutral-atom devices, the IQM device is not built as part of MQT Core itself.

## Installation

Install MQT Core with the {code}`iqm` extra to get Qiskit and the external {code}`iqm-qdmi` wheel in one step:

::::{tab-set}
:sync-group: installer

:::{tab-item} {code}`uv` _(recommended)_
:sync: uv

```console
$ uv pip install "mqt-core[iqm]"
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
(.venv) $ python -m pip install "mqt-core[iqm]"
```

:::
::::

If you prefer to work directly with the upstream IQM package, the {code}`iqm-qdmi[qiskit]` installation route remains supported as well.

## Usage

The IQM integration is exposed through {py:class}`~mqt.core.plugins.iqm.IQMBackend`.
This top-level plugin module keeps the IQM device integration separate from MQT Core's frontend-specific plugin packages while utilizing the upstream implementation and its packaged device library path export.
The backend is loaded explicitly, which leaves {py:class}`~mqt.core.plugins.qiskit.QDMIProvider` reserved for devices that are already visible through the FoMaC session.

```python
from mqt.core.plugins.iqm import IQMBackend
from qiskit import QuantumCircuit
from qiskit.compiler import transpile

backend = IQMBackend(
    base_url="https://resonance.iqm.tech",
    qc_alias="emerald:mock",
)

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

transpiled_qc = transpile(qc, backend)
result = backend.run(transpiled_qc, shots=128).result()
print(result.get_counts())
```

If you do not pass configuration explicitly, {py:class}`~mqt.core.plugins.iqm.IQMBackend` reads the same [environment variables as `iqm-qdmi`](https://iqm-finland.github.io/QDMI-on-IQM/usage.html):

- {code}`IQM_BASE_URL`
- {code}`IQM_TOKEN` or {code}`RESONANCE_API_KEY`
- {code}`IQM_TOKENS_FILE`
- {code}`IQM_QC_ID`
- {code}`IQM_QC_ALIAS`

The backend also exposes the same convenience helpers for binding Qiskit's sampler and estimator primitives:

```python
sampler = backend.sampler(default_shots=1024)
estimator = backend.estimator(default_precision=0.0)
```

## Relationship to the Upstream IQM Device Library

The {code}`iqm-qdmi` project ships the implementation in {code}`iqm.qdmi.qiskit`.
MQT Core exposes that backend through the module {py:mod}`mqt.core.plugins.iqm`.
See the [IQM QDMI Documentation](https://iqm-finland.github.io/QDMI-on-IQM/) for more details on authentication, device capabilities, and error handling.
