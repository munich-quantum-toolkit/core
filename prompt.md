We are building an MLIR based compiler for quantum computing that is currently solely scoped to the `mlir` subdirectory in the project and guarded behind a `BUILD_MQT_CORE_MLIR` CMake option. This option is on by default for regular builds, but currently disabled for the `mqt-core` Python package builds in @file:pyproject.toml

```toml
[tool.scikit-build.cmake.define]
BUILD_MQT_CORE_BINDINGS = "ON"
BUILD_MQT_CORE_TESTS = "OFF"
BUILD_MQT_CORE_SHARED_LIBS = "ON"
BUILD_MQT_CORE_MLIR = "OFF"
```

This was mainly because none of the functionality was actually exposed to Python, so there was little reason to enable it for the Python package builds.
Now we want to change that, and we want to actively distribute an entry point to the MLIR-based compiler to Python and make it accessible through the Python package.

This requires MLIR to be available while the package is being built; which boils down to `MLIR_DIR` being defined in a way that CMake can read it and find it.

- the reusable workflows we use in @file:ci.yml and @file:cd.yml already have options for enabling that also for the Python based tests and packaging workflows similar to the C++ workflows that already enable it. The action that is being run as part of that will populate an environment variable with the right path.
- for local developers, we want to make this as convenient as possible. People should not have to prefix their `uv` commands (like `uv sync`) with `MLIR_DIR=... <command>` so that the build succeeds. Either we can rely on environment files (`.env`) also from CMake or some other form of specifying the MLIR version to be used once in some place and the using that single source of truth everywhere. Whatever we end up doing here should also end up in the documentation of @file:installation.md in the "`## Setting Up MLIR`" section. We should stick to industry standards here as much as possible but strive for a pragmatic solution that does not cause friction for people.
- Changes will be needed for the linux cibuildwheel configuration in @file:pyproject.toml because LLVM will need to be installed as part of the container the build is running in. This should be possible via the `setup-mlir.sh` script that we provide and already document as part of the installation docs.

A key concern is that we want to keep the distribution as small as possible. Hence, we want to start with providing only a tiny wrapper around the CompilerPipeline in @file:CompilerPipeline.h and/or the mqt-cc executable in @file:mqt-cc.cpp .
The respective wrapper shall be placed in the `bindings` top level folder, similar to other bindings modules we have. It should be in a `mlir` subfolder in there and use nanobind to provide the respective bindings (very much inspired by the other bindings modules).
Essentially, the bindings may only expose a single `compile` function (although "compile" is a reserved keyword in python so a different name shall be used) that takes a Quantum Program and some options for the compilation, runs the compiler pipeline with these options and returns the result.
The programs that can go into the function should be either

- OpenQASM 3 text-based programs or files ending in `.qasm` that will be read into a `qc::QuantumComputation` similar to the `from_qasm_str` and `from_qasm` methods in @register_quantum_computation.cpp and will then be converted to MLIR using the translation in @TranslateQuantumComputationToQC.h.
- `QuantumComputation` class objects. so that our old frontend from Python can be used to get programs into the compiler. Just uses the translation like above
- `Qiskit QuantumCircuit`. Qiskit is optional and I want to support it as well. The blueprint here is how this is handled in @file:load.py . this can be called from the bindings module to parse the Qiskit circuit to a QuantumComputation and convert it to MLIR.
- A string based MLIR representation that is directly read into an MLIR module
- Optionally a binary MLIR module (but this will probably be tough to pass over the Python boundary) so maybe skip.
- A `jeff` module, which is a binary file. The best source for how to convert from a Jeff binary to our MLIR dialect is in @file:test_jeff_round_trip.cpp which shows deserialization and conversion to jeff mlir, to QCO, and that can be converted to QC, which is the input to the compiler pipeline.

I want to really make it seem like that one compiler entry point can accept all of the above, even when internally it needs to do quite a bit of patch work and routing to make this work. Parts of the patchwork will go away over time.
