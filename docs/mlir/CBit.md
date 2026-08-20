---
tocdepth: 3
---

```{include} Dialects/CBitDialect.md

```

## Memory lowering

Use `convert-cbit-to-memref` when a later pipeline requires generic memory. The
pass converts `!cbit.reg<N>` to `memref<Nxi1>` in operations, function
signatures, calls, branches, and returns. It lowers zero initialization to an
allocation followed by false stores. It lowers undefined initialization to an
allocation only.

This conversion is one-way. `mqt-cc` does not infer CBit semantics from an
arbitrary `memref<Nxi1>` because a memref does not record initialization or
public-result identity.
