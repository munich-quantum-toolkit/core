# Installed QDMI runtime deployment

Status: rebased on refreshed #2230; validation in progress.

## Motivation and scope

Installed CMake consumers need the same complete runtime layout as in-tree
applications. The existing mqt_copy_qdmi_runtime helper must stage imported
Client, driver, device libraries, manifests, provider assets and Windows DLLs.

This is Core PR #2231 on #2230, using QDMI #511 and current main APIs.

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

Validation is being repeated against current main.

Keep useful commits, human attribution and existing review threads. Do not
create archive branches or request reviews. Published artifacts require released
dependency pins.
