---
tocdepth: 3
---

```{include} Dialects/CBitDialect.md

```

## Register semantics

`!cbit.reg<N>` represents one non-aliasing register of `N` classical bits.
The width is static and must be positive. `cbit.alloc` states the initialization
of each register:

- `#cbit.init<zero>` defines every element as false.
- `#cbit.init<undefined>` leaves every element undefined until a `cbit.store`
  writes it. Reading an undefined element is undefined behavior.

`cbit.load` and `cbit.store` use `index` values and read or write `i1` values.
The verifier rejects constant indices outside the register width. Dynamic
indices remain valid and consumers decide which dynamic operations they can
support.

An optional source name on `cbit.alloc` preserves an input-language register
name. A source name does not make a register public. Only CBit registers
returned by the entry function are public result registers. This rule keeps
internal registers out of Qiskit and OpenQASM output.

The CBit dialect is shared by QC and QCO. Their conversions preserve register
identity, initialization, names, loads, stores, and returned results.

## Memory lowering

Use `convert-cbit-to-memref` when a later pipeline requires generic memory.
The pass converts `!cbit.reg<N>` to `memref<Nxi1>` in operations, function
signatures, calls, branches, and returns. It lowers zero initialization to an
allocation followed by false stores. It lowers undefined initialization to an
allocation only.

This conversion is one-way. MQT Core does not infer CBit semantics from an
arbitrary `memref<Nxi1>` because a memref does not record initialization or
public-result identity.
