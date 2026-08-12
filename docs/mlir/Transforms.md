---
tocdepth: 3
---

# Passes

Passes that belong to a single dialect are documented on that dialect's page:
{doc}`QC <QC>`, {doc}`QCO <QCO>`, and {doc}`QTensor <QTensor>`. The remaining
passes are documented here.

## Shared Passes

The following passes are not tied to a single dialect.

```{include} Passes/MQTTransforms.md

```

## QIR Passes

The following passes operate on modules that have already been lowered to the
LLVM dialect by the {doc}`QIR conversions <Conversions>`. They prepare such a
module for emission as QIR.

```{include} Passes/QIRTransforms.md

```
