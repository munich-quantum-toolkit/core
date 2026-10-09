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

Historical validation passed with GCC 13 and QDMI 1.4.0: the `release-no-mlir`
shared build and installation, configuration with `MQT_CORE_INSTALL=OFF`, and
repository lint. A separate installed-consumer probe checked that package
discovery preserved consumer settings and cache entries and that the installed
helper built a QDMI device with its colocated manifest. That probe is not part
of the repository test suite.
