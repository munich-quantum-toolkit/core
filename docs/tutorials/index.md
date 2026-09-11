# Tutorials

Learn how quantum compilation works by predicting a result, inspecting the
compiler's output, and changing the program. These six executable notebooks
assume basic Python and quantum computing knowledge; no MLIR experience is
needed.

1. **{doc}`compiler_representations`:** follow QC and QCO, remove redundant
   gates, and compare unitary matrices.
2. **{doc}`control_flow`:** trace registers, loops, and measurement-driven
   branches, then check their sampled results.
3. **{doc}`hardware_compilation`:** compare hardware constraints, inspect
   routing, and submit a compiled program through QDMI.
4. **{doc}`qir_execution`:** inspect Base and Adaptive QIR, execute feedback,
   and relate output records to counts.
5. **{doc}`qdmi_execution`:** discover device capabilities, submit and reuse
   jobs, and compare shots, counts, and simulator state results.
6. **{doc}`jeff_exchange`:** hand a structured program between compiler
   processes, preserve phase and loops, then compile it for execution.

For a first execution, use the {doc}`QPE walkthrough <../getting_started>`. The
{doc}`compiler guide <../mlir/mqt_compiler_collection>` describes interfaces and
options; these tutorials explain their effects through experiments.

## Run the notebooks

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Download
{download}`requirements.txt <requirements.txt>` and use the download link at the
start of each tutorial to save its notebook in the same directory.

In that directory, start JupyterLab with all tutorial dependencies:

```sh
uv run --no-project --with-requirements requirements.txt jupyter lab
```

`uv` manages and caches the environment. This command installs MQT Core,
JupyterLab, and the supported Qiskit visualization dependencies together. Open a
notebook with the Python 3 kernel and run its cells from top to bottom. Each
notebook has its own imports and parameters. See
[uv's Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/) to
use an existing project or an editor instead.

MQT Core's Python wheels include the compiler and local simulator; no separate
MLIR installation or hardware account is needed. See {doc}`../installation` for
platform requirements and source builds, and {doc}`../mlir/qiskit` for supported
Qiskit versions.

On this website, cells and figures show results from the documentation build.
Expand **Show code cell source** to inspect setup and plotting cells. These
cells remain in the downloaded notebooks; edit the parameter cells and rerun to
try the experiments.

```{toctree}
:maxdepth: 1
:hidden:

compiler_representations
control_flow
hardware_compilation
qir_execution
qdmi_execution
jeff_exchange
```
