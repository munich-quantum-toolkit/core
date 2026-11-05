# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Qiskit Backend.

Provides a Qiskit BackendV2-compatible interface to QDMI devices.
"""

from __future__ import annotations

__lazy_modules__ = {
    "math",
    "mqt.core.mlir",
    "mqt.core.plugins.qiskit.exceptions",
    "mqt.core.plugins.qiskit.job",
    "mqt.core.qdmi.builtin_driver",
    "numbers",
    "qiskit.circuit",
    "qiskit.circuit.library",
    "qiskit.primitives",
    "qiskit.transpiler",
}

from math import pi
from numbers import Integral
from typing import TYPE_CHECKING, Any

from qiskit.circuit import (
    Barrier,
    BreakLoopOp,
    ContinueLoopOp,
    ControlFlowOp,
    ControlledGate,
    ForLoopOp,
    Gate,
    IfElseOp,
    Parameter,
    QuantumCircuit,
    Store,
    SwitchCaseOp,
    WhileLoopOp,
)
from qiskit.circuit.library import (
    MCPhaseGate,
    MCXGate,
    get_standard_gate_name_mapping,
)
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit.providers import BackendV2, Options, QubitProperties
from qiskit.transpiler import InstructionProperties, Target

from mqt.core.mlir import (
    CompiledProgram,
    CompilerTarget,
    ProgramCapability,
    TargetEnvironment,
    compile_program,
    serialize_program,
)
from mqt.core.plugins.qiskit.exceptions import (
    CircuitValidationError,
    TranslationError,
    UnsupportedDeviceError,
    UnsupportedOperationError,
)
from mqt.core.plugins.qiskit.job import QDMIJob
from mqt.core.qdmi.builtin_driver import open_device

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Unpack

    from qiskit.circuit import Instruction
    from qiskit.circuit.parameterexpression import ParameterValueType
    from qiskit.transpiler import CouplingMap

    from mqt.core.mlir import CompilationOptions
    from mqt.core.plugins.qiskit.provider import QDMIProvider
    from mqt.core.qdmi import Device as QDMIDevice
    from mqt.core.typing import QDMIJobParameters, QDMISessionParameters, QiskitEstimatorOptions, QiskitSamplerOptions

    ParametersType = Mapping[Parameter, ParameterValueType] | Iterable[ParameterValueType]

__all__ = ["QDMIBackend"]


def __dir__() -> list[str]:
    return __all__


class QDMIBackend(BackendV2):
    """A Qiskit BackendV2 adapter for QDMI devices.

    This backend provides program submission to QDMI devices.
    It automatically introspects device capabilities and constructs a
    :class:`~qiskit.transpiler.Target` object with supported operations.

    Use :meth:`from_device_id` to open one client-visible device. Use
    :class:`~mqt.core.plugins.qiskit.provider.QDMIProvider` to enumerate
    client-visible devices.

    Args:
        device: QDMI device wrapper.
        provider: The provider instance that created this backend.

    Examples:
        Open a backend by stable device ID:

        >>> backend = QDMIBackend.from_device_id("mqt.ddsim.default")
    """

    def __init__(
        self,
        device: QDMIDevice,
        provider: QDMIProvider | None = None,
        *,
        device_id: str | None = None,
    ) -> None:
        """Initialize the backend with a QDMI device wrapper.

        Args:
            device: QDMI device wrapper.
            provider: Provider instance that created this backend.
            device_id: Stable ID for the opened device, if known.

        Raises:
            UnsupportedDeviceError: If the device cannot be represented in Qiskit's Target model.
        """
        super().__init__(name=device.name(), provider=provider, backend_version=device.version())
        self._device = device
        self._device_id = device_id
        try:
            self._environment = TargetEnvironment.from_device(device)
        except ValueError as exc:
            raise UnsupportedDeviceError(str(exc)) from exc
        self._native_width = max((site.id for site in self._environment.target.sites), default=-1) + 1
        self.last_job: QDMIJob | None = None
        self._target = self._build_target()

    @classmethod
    def from_device_id(
        cls,
        device_id: str,
        *,
        provider: QDMIProvider | None = None,
        **session_parameters: Unpack[QDMISessionParameters],
    ) -> QDMIBackend:
        """Open a client-visible QDMI device and adapt it for Qiskit.

        Args:
            device_id: Stable ID reported by the QDMI driver.
            provider: Provider to associate with the backend.
            session_parameters: Optional device-session parameters for the MQT Core QDMI driver.

        Returns:
            A Qiskit backend for a fresh QDMI device session.
        """
        return cls(
            device=open_device(device_id, **session_parameters),
            provider=provider,
            device_id=device_id,
        )

    @property
    def device(self) -> QDMIDevice:
        """The QDMI device the backend runs on."""
        return self._device

    @property
    def device_id(self) -> str | None:
        """Stable QDMI device ID, if known."""
        return self._device_id

    def sampler(self, **options: Unpack[QiskitSamplerOptions]) -> BackendSamplerV2:
        """Construct Qiskit's native sampler with typed keyword options.

        Returns:
            A sampler that executes on this backend.
        """
        return BackendSamplerV2(backend=self, options=dict(options))

    def estimator(self, **options: Unpack[QiskitEstimatorOptions]) -> BackendEstimatorV2:
        """Construct Qiskit's native estimator with typed keyword options.

        Returns:
            An estimator that executes on this backend.
        """
        return BackendEstimatorV2(backend=self, options=dict(options))

    @property
    def target(self) -> Target:
        """The Target describing the capabilities of the backend."""
        return self._target

    @property
    def provider(self) -> Any | None:  # ruff:ignore[any-type]
        """The provider that created the backend."""
        return self._provider

    @property
    def max_circuits(self) -> int | None:
        """The maximum number of circuits that can be run in a single job."""
        return None

    @property
    def options(self) -> Options:
        """The backend options."""
        return self._options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default backend options.

        Returns:
            Default options with shots=1024, memory=False, and max_retries=0.
        """
        return Options(shots=1024, memory=False, max_retries=0)

    def _job_parameters(self, options: Mapping[str, object]) -> QDMIJobParameters:  # ruff:ignore[no-self-use]
        """Validate and encode backend-specific execution options before submission.

        Subclasses declare supported options in :meth:`_default_options` and
        override this hook to map their values to QDMI custom job parameters.
        An override replaces this implementation and must handle every option
        it declares. The mapping contains resolved backend defaults and per-run
        overrides, excluding the base options ``shots``, ``memory``, and
        ``max_retries``. This hook runs once, before any job is submitted, and
        must not submit jobs or mutate backend options.

        Args:
            options: Effective execution options for every circuit in this run.

        Returns:
            Custom keyword arguments for :meth:`~mqt.core.qdmi.Device.submit_job`.

        Raises:
            CircuitValidationError: If an option has no submission mapping.
        """
        if options:
            msg = f"Unsupported execution options: {', '.join(sorted(options))}"
            raise CircuitValidationError(msg)
        return {}

    def _build_target(self) -> Target:
        """Project the shared device snapshot into Qiskit's physical positions.

        Returns:
            The placeable native operations and their calibration.

        Raises:
            UnsupportedDeviceError: If no computational sites remain.
            UnsupportedOperationError: If Qiskit cannot represent a capability.
        """
        snapshot = self._environment.target
        self._placeable_sites = set(snapshot.placeable_sites)
        if not self._placeable_sites:
            msg = "The device has no sites with single-qubit gates and measurement."
            raise UnsupportedDeviceError(msg)
        width = max(self._placeable_sites) + 1
        target = Target(description=f"QDMI device: {self.name}", num_qubits=width)
        standard: dict[str, Instruction | type[Instruction]] = get_standard_gate_name_mapping()
        standard.update({"gphase": standard["global_phase"], "mcx": MCXGate, "mcp": MCPhaseGate})
        variants: dict[
            tuple[str, int | None, int, tuple[float | None, ...], tuple[tuple[float, float] | None, ...]],
            tuple[Instruction | type[Instruction], dict[tuple[int, ...] | None, InstructionProperties], str],
        ] = {}
        self._operation_names: dict[str, str] = {}
        reserved_names = {operation.canonical_name for operation in snapshot.operations}
        used_names: set[str] = set()
        projections: list[tuple[CompilerTarget.OperationCapability, Instruction | type[Instruction], str]] = []
        for capability in snapshot.operations:
            name = capability.canonical_name
            if name == "barrier":
                continue
            gate = standard.get(name)
            variadic = capability.arity.kind == CompilerTarget.OperationArityKind.VARIADIC
            if gate is None or (isinstance(gate, type) and not variadic):
                if variadic:
                    msg = f"Qiskit cannot represent variadic native operation {capability.name!r}."
                    raise UnsupportedOperationError(msg)
                gate = None
            elif not isinstance(gate, type) and (
                gate.num_qubits != capability.arity.value or len(gate.params) != capability.num_parameters
            ):
                gate = None
            if gate is None:
                gate = Gate(
                    name, capability.arity.value, [Parameter(f"p{i}") for i in range(capability.num_parameters)]
                )
            if capability.name in {"gpi", "gpi2"} and capability.canonical_name == "r":
                gate = Gate(capability.name, 1, [Parameter("phi")])
                definition = QuantumCircuit(1, global_phase=pi / 2 if capability.name == "gpi" else 0)
                definition.r(capability.fixed_parameters[0], gate.params[0], 0)
                gate.definition = definition
            name = ("mcphase" if name == "mcp" else name) if isinstance(gate, type) else gate.name
            self._operation_names[capability.name] = name
            projections.append((capability, gate, name))
            if variadic and not isinstance(gate, type):
                projections.extend(
                    (capability, controlled, controlled.name)
                    for controlled in standard.values()
                    if isinstance(controlled, ControlledGate)
                    and controlled.base_gate is not None
                    and controlled.base_gate.name == gate.name
                    and len(controlled.params) == capability.num_parameters
                    and controlled.ctrl_state == (1 << controlled.num_ctrl_qubits) - 1
                )
                if name in {"x", "p"}:
                    projections.append((
                        capability,
                        MCXGate if name == "x" else MCPhaseGate,
                        "mcx" if name == "x" else "mcphase",
                    ))
        projections.sort(
            key=lambda item: any(value is not None for value in (*item[0].fixed_parameters, *item[0].parameter_bounds))
        )
        variant_counts: dict[str, int] = {}
        for capability, projected_gate, name in projections:
            gate = projected_gate
            fixed = capability.fixed_parameters
            bounds = capability.parameter_bounds
            if name in {"gpi", "gpi2"} and capability.canonical_name == "r":
                fixed = fixed[1:]
                bounds = bounds[1:]
            key = (
                name,
                None if isinstance(gate, type) else gate.num_qubits,
                capability.num_parameters,
                tuple(fixed),
                tuple(bounds),
            )
            if key not in variants:
                if not isinstance(gate, type):
                    gate = gate.copy()
                    if fixed:
                        gate.params = [
                            value if value is not None else parameter
                            for parameter, value in zip(gate.params, fixed, strict=True)
                        ]
                variant = name
                if variant in used_names:
                    suffix = variant_counts.get(name, 1)
                    variant = f"{name}_{suffix}"
                    while variant in reserved_names or variant in used_names:
                        suffix += 1
                        variant = f"{name}_{suffix}"
                    variant_counts[name] = suffix + 1
                used_names.add(variant)
                variants[key] = (gate, {}, variant)
            _, properties, _ = variants[key]
            placements = capability.site_tuples
            if placements:
                loci = [
                    (tuple(item.sites), item.duration, item.fidelity)
                    for item in placements
                    if set(item.sites) <= self._placeable_sites
                ]
            else:
                loci = [(None, None, None)]
            for locus, duration, fidelity in loci:
                seconds = self._duration_seconds(capability.duration if duration is None else duration)
                effective_fidelity = capability.fidelity if fidelity is None else fidelity
                properties.setdefault(
                    locus,
                    InstructionProperties(
                        duration=seconds, error=None if effective_fidelity is None else 1.0 - effective_fidelity
                    ),
                )
        for (_, _, _, _, bounds), (gate, properties, name) in variants.items():
            if not properties:
                continue
            if isinstance(gate, type):
                target.add_instruction(gate, name=name)
            else:
                target.add_instruction(
                    gate,
                    {None: properties[None]} if None in properties else properties,
                    name=name,
                    angle_bounds=list(bounds) or None,
                )
        capabilities = {capability.capability_id for capability in self._environment.payload_specification.capabilities}
        for capability, name, instruction in (
            (ProgramCapability.FORWARD_BRANCHING, "if_else", IfElseOp),
            (ProgramCapability.COUNTED_ITERATION, "for_loop", ForLoopOp),
            (ProgramCapability.CONDITIONAL_LOOP, "while_loop", WhileLoopOp),
            (ProgramCapability.CONDITIONAL_LOOP, "break_loop", BreakLoopOp),
            (ProgramCapability.CONDITIONAL_LOOP, "continue_loop", ContinueLoopOp),
            (ProgramCapability.MULTIWAY_BRANCHING, "switch_case", SwitchCaseOp),
        ):
            if capability in capabilities:
                target.add_instruction(instruction, name=name)
        if ProgramCapability.CLASSICAL_ASSIGNMENT in capabilities:
            target.add_instruction(Store, name="store")
        target.add_instruction(Barrier, name="barrier")
        if any(site.t1 is not None or site.t2 is not None for site in snapshot.sites):
            properties = [QubitProperties() for _ in range(width)]
            for site in snapshot.sites:
                if site.id in self._placeable_sites:
                    properties[site.id] = QubitProperties(
                        t1=self._duration_seconds(site.t1), t2=self._duration_seconds(site.t2)
                    )
            target.qubit_properties = properties
        self._target_names: dict[str, list[str]] = {}
        self._target_classes: list[tuple[type[Instruction], str]] = []
        for name in target.operation_names:
            operation = target.operation_from_name(name)
            if isinstance(operation, type):
                self._target_classes.append((operation, name))
                self._target_names.setdefault(name, []).append(name)
            else:
                self._target_names.setdefault(operation.name, []).append(name)
        return target

    def _duration_seconds(self, duration: int | None) -> float | None:
        """Convert snapshot timing metadata to seconds.

        Returns:
            The duration in seconds, or None for absent metadata.

        Raises:
            UnsupportedOperationError: If the reported unit is unsupported.
        """
        if duration is None:
            return None
        unit = self._environment.target.duration_unit
        seconds_per_unit = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15}
        if unit is None or unit.unit not in seconds_per_unit:
            msg = "Cannot convert device timing metadata to seconds without a supported duration unit."
            raise UnsupportedOperationError(msg)
        return duration * unit.scale_factor * seconds_per_unit[unit.unit]

    @property
    def coupling_map(self) -> CouplingMap | None:
        """Explicit coupling constraints, without expanding all-to-all devices."""
        if self._environment.target.connectivity_kind == CompilerTarget.ConnectivityKind.ALL_TO_ALL:
            return None
        return super().coupling_map

    def validate_circuit(self, circuit: QuantumCircuit, *, native: bool = False) -> None:
        """Check operations and native placements against the snapshot target.

        Args:
            circuit: Native or logical circuit to check.
            native: Also check physical sites and parameter values.

        Raises:
            CircuitValidationError: If the circuit exceeds the device width.
            UnsupportedOperationError: If an instruction is not supported.
        """
        if native and circuit.num_qubits > self._native_width:
            msg = (
                f"Circuit has {circuit.num_qubits} qubits, "
                f"but the native device has {self._native_width} addressable positions."
            )
            raise CircuitValidationError(msg)
        pending = [(circuit, tuple(range(circuit.num_qubits)))]
        while pending:
            block, indices = pending.pop()
            for instruction in block.data:
                operation = instruction.operation
                if operation.name == "barrier":
                    continue
                qargs = tuple(indices[block.find_bit(bit).index] for bit in instruction.qubits)
                if isinstance(operation, ControlFlowOp):
                    if operation.name not in self._target.operation_names:
                        msg = f"Unsupported control flow operation: '{operation.name}'"
                        raise UnsupportedOperationError(msg)
                    pending.extend((body, qargs) for body in reversed(operation.blocks))
                    continue
                operation_name = self._operation_names.get(operation.name, operation.name)
                names = self._target_names.get(operation_name)
                if names is None:
                    names = [name for gate, name in self._target_classes if isinstance(operation, gate)]
                supported = (not native or set(qargs) <= self._placeable_sites) and any(
                    self._target.instruction_supported(
                        name, **({"qargs": qargs, "parameters": operation.params} if native else {})
                    )
                    for name in names
                )
                if not supported:
                    msg = (
                        f"Operation '{operation.name}' with parameters {operation.params} "
                        f"is not advertised on native device qubits {qargs}."
                    )
                    msg += " Use backend.compile to compile and place logical circuits."
                    raise UnsupportedOperationError(msg)

    def _preprocess_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:  # ruff:ignore[no-self-use]
        """Rewrite a bound circuit before validation and conversion.

        Called once per circuit in :meth:`run`, after parameter binding and
        before operation-support validation and program conversion.
        Subclasses may override this to enforce receiver-specific input
        requirements. The default implementation returns the circuit unchanged.

        Args:
            circuit: The bound circuit to preprocess.

        Returns:
            The (possibly rewritten) circuit to use for validation and conversion.
        """
        return circuit

    def compile(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        *,
        options: CompilationOptions | None = None,
    ) -> CompiledProgram | list[CompiledProgram]:
        """Compile circuits for this backend's native payload and result layout.

        Returns:
            One compiled program, or an ordered list for a sequence.
        """
        circuits = (
            self._preprocess_circuit(circuits)
            if isinstance(circuits, QuantumCircuit)
            else [self._preprocess_circuit(circuit) for circuit in circuits]
        )
        return compile_program(circuits, target=self, options=options)

    def run(
        self,
        run_input: QuantumCircuit | CompiledProgram | Sequence[QuantumCircuit | CompiledProgram],
        parameter_values: Sequence[ParametersType] | None = None,
        **options: Any,  # ruff:ignore[any-type]
    ) -> QDMIJob:
        """Submit native circuits without running compilation or routing.

        Use ``backend.run(backend.compile(circuit))`` when compilation is needed.
        Compiled payloads are submitted unchanged. Serialization
        can normalize terminal measurements and retain their output mapping;
        it does not change the compiled quantum operations or placement.

        Args:
            run_input: Native circuits or compiled programs, individually or in a sequence.
            parameter_values: Optional parameter values to bind to the circuits. If provided, must be a sequence
                with one entry per circuit. Each entry can be either a dictionary mapping parameters to values,
                or a sequence of values in the order of circuit.parameters.
            **options: Execution options: nonnegative integer ``shots`` and ``max_retries``, and boolean ``memory``.
                Memory requires genuine QDMI SHOTS results. Simulator seeds are unsupported.
                Subclasses may declare additional options and encode them in ``_job_parameters``.

        Returns:
            Job handle for the execution. For multiple circuits, the job aggregates results from all circuits.

        Raises:
            CircuitValidationError: If circuit validation fails (e.g., invalid shots, unbound parameters,
                parameter_values length mismatch).
            UnsupportedOperationError: If a circuit contains unsupported operations.
            JobSubmissionError: If job submission to the device fails.
            TranslationError: If native payload serialization fails.

        Examples:
            Run a single circuit with parameter values:

            >>> from qiskit.circuit import Parameter, QuantumCircuit
            >>> theta = Parameter("theta")
            >>> qc = QuantumCircuit(1)
            >>> qc.ry(theta, 0)
            >>> qc.measure_all()
            >>> job = backend.run(qc, parameter_values=[{theta: 1.5708}])

            Run multiple circuits with different parameter values:

            >>> qc1 = QuantumCircuit(1)
            >>> qc1.ry(theta, 0)
            >>> qc1.measure_all()
            >>> qc2 = QuantumCircuit(1)
            >>> qc2.ry(theta, 0)
            >>> qc2.measure_all()
            >>> job = backend.run([qc1, qc2], parameter_values=[{theta: 0.5}, {theta: 1.5}])
        """  # ruff:ignore[docstring-extraneous-exception] The validation helper raises operation errors.
        self.last_job = None
        circuits = [run_input] if isinstance(run_input, (QuantumCircuit, CompiledProgram)) else run_input

        if not circuits:
            msg = "No circuits provided to run. At least one circuit is required."
            raise CircuitValidationError(msg)

        if parameter_values is not None and len(parameter_values) != len(circuits):
            msg = (
                f"Length of parameter_values ({len(parameter_values)}) must match "
                f"the number of circuits ({len(circuits)})"
            )
            raise CircuitValidationError(msg)

        # Native primitives pass an unset simulator seed to every backend.
        if options.get("seed_simulator") is None:
            options.pop("seed_simulator", None)
        if unsupported := options.keys() - self._options.keys():
            msg = f"Unsupported execution options: {', '.join(sorted(unsupported))}"
            raise CircuitValidationError(msg)

        shots_opt = options.get("shots", self._options.shots)
        if not isinstance(shots_opt, Integral) or isinstance(shots_opt, bool):
            msg = f"Invalid 'shots' value: {shots_opt!r}"
            raise CircuitValidationError(msg)
        shots = int(shots_opt)
        if shots < 0:
            msg = f"'shots' must be >= 0, got {shots}"
            raise CircuitValidationError(msg)
        memory = options.get("memory", self._options.memory)
        if not isinstance(memory, bool):
            msg = f"Invalid 'memory' value: {memory!r}"
            raise CircuitValidationError(msg)

        max_retries = options.get("max_retries", self._options.max_retries)
        if isinstance(max_retries, bool) or not isinstance(max_retries, Integral) or max_retries < 0:
            msg = f"max_retries must be a nonnegative integer, got {max_retries!r}."
            raise CircuitValidationError(msg)
        max_retries = int(max_retries)
        effective_options = {**dict(self._options), **options}
        job_parameters = self._job_parameters({
            name: value for name, value in effective_options.items() if name not in {"shots", "memory", "max_retries"}
        })
        prepared_programs: list[CompiledProgram] = []
        # Prepare every circuit before submitting any job, so validation cannot leave a partial batch.

        for idx, circuit in enumerate(circuits):
            if isinstance(circuit, CompiledProgram):
                if parameter_values is not None and parameter_values[idx] is not None:
                    msg = "Parameter bindings do not apply to compiled programs."
                    raise CircuitValidationError(msg)
                prepared_programs.append(circuit)
                continue
            if not isinstance(circuit, QuantumCircuit):
                msg = "run accepts QuantumCircuit or CompiledProgram inputs."
                raise CircuitValidationError(msg)
            bound_circuit = circuit
            if parameter_values is not None:
                try:
                    bound_circuit = circuit.assign_parameters(parameter_values[idx])
                    bound_circuit.name = circuit.name
                except Exception as exc:
                    msg = f"Failed to bind parameters for circuit {idx}: {exc}"
                    raise CircuitValidationError(msg) from exc

            # Validate circuit has no unbound parameters
            if bound_circuit.parameters:
                params = ", ".join(sorted(p.name for p in bound_circuit.parameters))
                msg = (
                    f"Circuit contains unbound parameters: {params}. Provide `parameter_values` or bind them manually."
                )
                raise CircuitValidationError(msg)

            bound_circuit = self._preprocess_circuit(bound_circuit)
            self.validate_circuit(bound_circuit, native=True)

            try:
                prepared_programs.append(serialize_program(bound_circuit, target=self._environment))
            except Exception as exc:
                msg = f"Failed to serialize circuit {idx}: {exc}"
                raise TranslationError(msg) from exc

        job = QDMIJob(
            self,
            prepared_programs,
            shots=shots,
            memory=memory,
            max_retries=max_retries,
            job_parameters=job_parameters,
        )
        self.last_job = job
        job.submit()
        return job
