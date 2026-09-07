# MQT Core MLIR agent guide

This file applies to `mlir/`. Read the root `AGENTS.md` first. The canonical
policy is [`docs/mlir/development.md`](../docs/mlir/development.md) and is
normative. This file is its short, scoped routing layer.

## Core rules

- Never add `const` to `Value` or its typed, result, and block-argument forms;
  range views; `Operation`, `Block`, `Region`, `ModuleOp`; or a typed operation
  wrapper, including through `const auto`. Copy these cheap handles and views.
- Do not add top-level `const` to any by-value parameter.
- A pass must not crash on valid IR and its successful output must verify.
- Trace the contract from the frontend or builder through verifiers, interfaces,
  transforms, and consumers before changing IR. TableGen alone does not specify
  every semantic or pipeline requirement.
- Give each invariant one owning layer. Verifiers check their own operation;
  conversions and exporters check their supported subset. Diagnose unsupported
  valid IR instead of asserting, silently widening support, or emitting partial
  success. Failed rewrite matches must leave IR unchanged.
- Preserve deterministic output; never expose pointer or unordered traversal
  order.
- Search upstream MLIR before adding an MQT-specific operation, interface,
  trait, conversion, or utility.
- Prefer existing folding, canonicalization, and analysis facilities when they
  satisfy the contract. Retain custom state only for a concrete correctness or
  complexity requirement, and test the relevant boundary.
- Use direct GoogleTest/CTest tests. Do not add `lit` or FileCheck.
- Assert semantics and required normal forms. Exact SSA trees, target choices,
  text, and operation counts need a stated reason, such as a consumer contract,
  determinism, or a complexity bound. Preserve phase, wire identity, numerical
  limits, and negative cases when replacing a representation-specific oracle.

## Load detailed guidance when relevant

- Before changing IR APIs, passes, verifiers, rewrites, or diagnostics, read the
  corresponding sections of the canonical policy.
- Before changing tests, debugging a failure, or proposing a performance
  rewrite, read its testing, debugging, or performance section.

## Maintenance

- Review this guide and `mlir/.clang-tidy` on every LLVM/MLIR major upgrade.
- Existing code is evidence, not authority, when it conflicts with current
  policy.
