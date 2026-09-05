# Installed QDMI runtime deployment

Status: independently rebased and locally validated; hosted Windows CI pending.

## Motivation and scope

Installed CMake consumers need the same complete runtime layout as in-tree
applications. The existing mqt_copy_qdmi_runtime helper must stage imported
Client, driver, device libraries, manifests, provider assets and Windows DLLs.

This is Core PR #2231 on #2230, targeting Core 4.1 / QDMI 1.4. It does not
depend on metadata removal, batching, or payload capabilities. No payload-format
header is introduced by the driver workstream.

## Decisions

Reuse the existing imported-device fixture as a real find_package consumer. Use
copy_if_different for local and imported runtime targets. Imported targets must
not become build dependencies. For Windows, retain the non-imported
linker-language-bearing closure used to compute transitive imported DLLs.
Preserve device metadata and asset copying, and use the build RPATH while
running staged build-tree applications.

This changes deployment only. It does not add Client APIs, providers, compiler
behavior, or a second package-consumer harness.

## Validation

Run the release build, both imported-device fixture tests, and the full native
suite. The fixture must resolve installed Core targets, execute the consumer,
and compare staged libraries, manifest, assets and Windows dependency files.
Check that the helper disables BUILD_WITH_INSTALL_RPATH on its consumer. Run
repository lint; Windows hosted CI remains necessary for real DLL loading.

The release build and native suite pass: 3,873 tests pass and one existing
optional-device test skips. Both installed-consumer fixture tests pass.

Keep useful commits, human attribution and existing review threads. Do not
create archive branches or request reviews. Published artifacts require released
dependency pins.
