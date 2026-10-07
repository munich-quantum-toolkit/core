# Isolate installed CMake package policy

Status: complete.

## Goal and scope

`find_package(mqt-core)` imports targets and explicit helper commands without
changing the consumer's build policy. The root source build owns cache
launchers, in-source build prevention, install directories, and project
defaults.

The boundary is `cmake/mqt-core-config.cmake.in`, the installed file list in
`src/CMakeLists.txt`, and `cmake/AddMQTQDMIDevice.cmake`. Keep the exported
`MQT::ProjectOptions` and `MQT::ProjectWarnings` targets used by the Python and
test helpers. Compiler defaults, binding optimization, and DD ABI changes are
outside this task.

## Decisions

Remove the policy modules from the installed config and install list. Local
downstream and documentation searches found no explicit use of these installed
modules. The retained helpers use absolute or built-in includes, so the config
does not need to append to `CMAKE_MODULE_PATH`.

Load `GNUInstallDirs` when `mqt_configure_qdmi_device` is called. The source
root loads it explicitly because its own RPATH and output-directory setup needs
those defaults before any helper call.

## Validation

`test/cmake/installed_consumer/CMakeLists.txt` is one standalone regression. Run
it against an installed prefix, then build its device target:

```console
cmake -S test/cmake/installed_consumer -B build/installed-consumer -DCMAKE_PREFIX_PATH=<prefix>
cmake --build build/installed-consumer
```

It checks that package discovery preserves consumer settings and their cache
entries, including unset defaults, and builds a QDMI device with its colocated
manifest through the installed helper. The regression failed against the prior
installed config and passed after the change with QDMI 1.4.0.

Local validation passed: the `release-no-mlir` shared build and installation
with GCC 13, configuration with `MQT_CORE_INSTALL=OFF`, the installed consumer's
configuration and build, and repository lint. Full compiler and wheel
qualification belongs to the separate release validation work.
