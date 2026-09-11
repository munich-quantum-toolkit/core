# MQT Core MLIR binding agent guide

Read the root `AGENTS.md`, then follow [`mlir/AGENTS.md`](../../mlir/AGENTS.md)
and the canonical
[`MLIR development policy`](../../docs/mlir/development.md).

- Apply the MLIR no-`const` rule to bound MLIR handles and typed operation
  wrappers.
- Preserve the supported Python API unless a breaking change is explicitly
  approved.
- Regenerate and validate type stubs after every binding change. Never edit
  generated `.pyi` files by hand.
- Keep Python imports lazy where the existing binding does so for an optional
  integration.
