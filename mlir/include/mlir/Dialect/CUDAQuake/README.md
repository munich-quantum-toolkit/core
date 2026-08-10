# CUDA-Q compatibility dialect provenance

The definitions in this directory are a deliberately small compatibility
projection of the textual `quake` and `cc` MLIR dialects from NVIDIA CUDA-Q
0.15.0, commit `84c2f5bf9d2911d7d14c9e3251843832bbc12843`.

The assembly formats and the small custom parser/printer portions were derived
from CUDA-Q's `QuakeTypes.td`, `QuakeOps.td`, `CCTypes.td`, `CCOps.td`, and the
corresponding implementation files. Those portions are licensed under the Apache
License 2.0 and retain NVIDIA's copyright notices. MQT-specific conversion and
compiler integration code is not derived from CUDA-Q and uses MQT Core's MIT
license.

This is not a full copy of either CUDA-Q dialect. New definitions should only be
added when they occur at the supported textual interoperability boundary.

See `LICENSE` in this directory for the Apache License 2.0 text.
