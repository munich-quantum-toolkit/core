# OQ3 Dialect

The OQ3 dialect is an experimental typed semantic representation for OpenQASM
3 programs. It preserves OpenQASM-specific concepts, such as source-level gate
definitions, ordered modifiers, and inclusive ranges, while ordinary classical
computation uses the builtin MLIR dialects.

```{warning}
OQ3 is internal and experimental. Its textual representation is not a stable
public interface.
```

```{include} Dialects/OQ3Dialect.md
```

## Passes

```{include} Passes/OQ3Transforms.md
```
