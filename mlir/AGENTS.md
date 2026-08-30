# MQT Core MLIR agent guide

This file applies to `mlir/`. Read the root `AGENTS.md` first. The canonical
policy is [`docs/mlir/development.md`](../docs/mlir/development.md) and is
normative. This file is only its short, always-loaded routing layer.

## Always-loaded rules

- Never add `const` to `Value` or its typed, result, and block-argument forms;
  range views; `Operation`, `Block`, `Region`, `ModuleOp`; or a typed operation
  wrapper, including through `const auto`. Copy these cheap handles and views.
- Do not add top-level `const` to any by-value parameter.
- A pass must not crash on valid IR and its successful output must verify.
- Preserve deterministic output; never expose pointer or unordered traversal
  order.
- Search upstream MLIR before adding an MQT-specific operation, interface,
  trait, conversion, or utility.
- Use direct GoogleTest/CTest tests. Do not add `lit` or FileCheck.

## Load detailed guidance when relevant

- Before changing IR APIs, passes, verifiers, rewrites, or diagnostics, read the
  corresponding sections of the canonical policy.
- Before changing tests, debugging a failure, or proposing a performance
  rewrite, read its testing, debugging, or performance section.

## Maintenance

- Review this guide and `mlir/.clang-tidy` on every LLVM/MLIR major upgrade.
- Existing code is evidence, not authority, when it conflicts with current
  policy.
