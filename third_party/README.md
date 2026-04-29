# Third-Party Dependencies

This directory contains vendored third-party dependencies that are shipped as
part of MQT Core.

## Layout

Each vendored dependency should follow this structure:

- `third_party/<name>/`
- `third_party/<name>/CMakeLists.txt`
- `third_party/<name>/METADATA.yml`
- `third_party/<name>/LICENSE*`
- Upstream sources (minimized to what MQT Core requires)

## Policy

- Keep vendored sources unmodified whenever possible.
- If patches are required, document them in `METADATA.yml`.
- Preserve upstream license files.
- Keep the dependency version and source URL in `METADATA.yml`.
- Exclude vendored sources from local formatters/linters.

## Current dependencies

- `nlohmann_json` (single-include headers)
- `spdlog` (headers + compiled library sources)
- `boost_mp` (Boost.Multiprecision headers)
