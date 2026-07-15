# QDMI configuration

QDMI configuration is versioned and trusted: discovering it only parses device
definitions, while opening a configured library executes native code.

```json
{
  "schema-version": 1,
  "qdmi": {
    "devices": [
      {
        "id": "example.device",
        "library": "libexample-device.so",
        "prefix": "EXAMPLE",
        "enabled": true,
        "session": {
          "base-url": "https://device.example",
          "auth-file": "credentials.json",
          "custom1": "provider-specific"
        }
      }
    ]
  }
}
```

Each enabled QDMI definition requires a unique `id`, `library`, and `prefix`.
Relative library and authentication-file paths are resolved against the file
that declared them. Unknown keys, invalid types, duplicate IDs within one
source, and incomplete enabled definitions are errors with source and
property-path diagnostics.

Sources are merged field-by-field by ID in this order:

1. generated manifest fragments packaged beside the device libraries;
2. the system `mqt-core.json`;
3. the user/XDG `mqt-core.json`;
4. the nearest project `mqt-core.json`, or `[tool.mqt-core.qdmi]` in
   `pyproject.toml` when no dedicated file exists in that directory;
5. `MQT_CORE_QDMI_CONFIG_JSON`;
6. constructor/runtime overrides.

`MQT_CORE_QDMI_CONFIG_FILE` or `ConfigOptions.explicitFile` replaces the system,
user, and project levels. Packaged definitions remain available unless
`isolated` is enabled. Setting `enabled` to `false` masks an inherited
definition.

For a static C++ consumer, pass the directory containing generated `*.qdmi.json`
fragments explicitly because a fully static executable has no portable module
origin:

```cpp
qdmi::ConfigOptions options;
options.configRoot = "/opt/site/lib";
qdmi::DeviceManager manager(options);
```

Runtime registration is also available without modifying configuration files:

```cpp
manager.registerDevice({.id = "site.device",
                        .library = "/opt/site/lib/libdevice.so",
                        .prefix = "SITE"});
auto device = manager.open("site.device");
```
