# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed benchmark instances and analytic references."""

import enum
import fractions
from collections.abc import Mapping

import mqt.core.mlir

class Output:
    """One logical classical output register."""

    @property
    def name(self) -> str:
        """The register name."""

    @property
    def width(self) -> int:
        """The number of big-endian outcome bits."""

class Evaluation:
    """The result of comparing counts with a reference."""

    @property
    def total_variation_distance(self) -> float:
        """The total variation distance from the ideal distribution."""

    @property
    def squared_hellinger_fidelity(self) -> float:
        """The squared Hellinger fidelity with the ideal distribution."""

    @property
    def success_probability(self) -> float | None:
        """The observed success probability, when defined."""

class BVMethod(enum.Enum):
    """Static allocation or dynamic measurement and qubit reuse."""

    STATIC = 0

    DYNAMIC = 1

class BVOptions:
    """Parameters for a Bernstein--Vazirani benchmark."""

    def __init__(self, *, hidden_bitstring: str, method: BVMethod = BVMethod.STATIC) -> None: ...
    @property
    def hidden_bitstring(self) -> str:
        """The big-endian hidden bitstring."""

    @property
    def method(self) -> BVMethod:
        """The circuit method."""

class BV:
    """A validated Bernstein--Vazirani benchmark."""

    def __init__(self, options: BVOptions) -> None: ...
    @property
    def options(self) -> BVOptions:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> Output:
        """The logical output register."""

    def probability(self, outcome: str) -> float:
        """Return the ideal probability of an outcome."""

    def evaluate(self, counts: Mapping[str, int]) -> Evaluation:
        """Compare sampled counts with the ideal distribution."""

    def generate(self) -> mqt.core.mlir.QCProgram:
        """Generate the benchmark as a QC program."""

    @property
    def request_json(self) -> str:
        """The canonical request JSON."""

    @property
    def manifest_json(self) -> str:
        """The canonical manifest JSON."""

    @property
    def case_id(self) -> str:
        """The stable semantic case ID."""

    @staticmethod
    def from_request_json(json: str, *, source: str = "<request>") -> BV:
        """Parse a strict benchmark request."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> BV:
        """Parse a strict benchmark manifest."""

class GHZTopology(enum.Enum):
    """Entangling topology for GHZ preparation."""

    LINEAR = 0

    STAR = 1

class GHZBasis(enum.Enum):
    """Measurement basis for GHZ verification."""

    Z = 0

    X = 1

class GHZOptions:
    """Parameters for a GHZ benchmark."""

    def __init__(
        self, *, qubits: int, topology: GHZTopology = GHZTopology.LINEAR, basis: GHZBasis = GHZBasis.Z
    ) -> None: ...
    @property
    def qubits(self) -> int:
        """The number of qubits."""

    @property
    def topology(self) -> GHZTopology:
        """The entangling topology."""

    @property
    def basis(self) -> GHZBasis:
        """The measurement basis."""

class GHZ:
    """A validated GHZ benchmark."""

    def __init__(self, options: GHZOptions) -> None: ...
    @property
    def options(self) -> GHZOptions:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> Output:
        """The logical output register."""

    def probability(self, outcome: str) -> float:
        """Return the ideal probability of an outcome."""

    def evaluate(self, counts: Mapping[str, int]) -> Evaluation:
        """Compare sampled counts with the ideal distribution."""

    def generate(self) -> mqt.core.mlir.QCProgram:
        """Generate the benchmark as a QC program."""

    @property
    def request_json(self) -> str:
        """The canonical request JSON."""

    @property
    def manifest_json(self) -> str:
        """The canonical manifest JSON."""

    @property
    def case_id(self) -> str:
        """The stable semantic case ID."""

    @staticmethod
    def from_request_json(json: str, *, source: str = "<request>") -> GHZ:
        """Parse a strict benchmark request."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> GHZ:
        """Parse a strict benchmark manifest."""

class GroverOptions:
    """Parameters for a Grover benchmark."""

    def __init__(self, *, marked_bitstring: str, iterations: int | None = None) -> None: ...
    @property
    def marked_bitstring(self) -> str:
        """The big-endian marked outcome."""

    @property
    def iterations(self) -> int | None:
        """The iteration count, or ``None`` for automatic selection."""

class Grover:
    """A validated single-solution Grover benchmark."""

    def __init__(self, options: GroverOptions) -> None: ...
    @property
    def options(self) -> GroverOptions:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> Output:
        """The logical output register."""

    @property
    def qubits(self) -> int:
        """The number of search qubits."""

    def probability(self, outcome: str) -> float:
        """Return the ideal probability of an outcome."""

    def evaluate(self, counts: Mapping[str, int]) -> Evaluation:
        """Compare sampled counts with the ideal distribution."""

    def generate(self) -> mqt.core.mlir.QCProgram:
        """Generate the benchmark as a QC program."""

    @property
    def request_json(self) -> str:
        """The canonical request JSON."""

    @property
    def manifest_json(self) -> str:
        """The canonical manifest JSON."""

    @property
    def case_id(self) -> str:
        """The stable semantic case ID."""

    @staticmethod
    def from_request_json(json: str, *, source: str = "<request>") -> Grover:
        """Parse a strict benchmark request."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> Grover:
        """Parse a strict benchmark manifest."""

class QFTMethod(enum.Enum):
    """Full-register or semiclassical measurement-and-feed-forward method."""

    STANDARD = 0

    SEMICLASSICAL = 1

class QFTOptions:
    """Parameters for a QFT benchmark."""

    def __init__(self, *, qubits: int, period_exponent: int, method: QFTMethod = QFTMethod.STANDARD) -> None: ...
    @property
    def qubits(self) -> int:
        """The number of transformed qubits."""

    @property
    def period_exponent(self) -> int:
        """The base-two input-period exponent."""

    @property
    def method(self) -> QFTMethod:
        """The circuit method."""

class QFT:
    """A validated QFT benchmark."""

    def __init__(self, options: QFTOptions) -> None: ...
    @property
    def options(self) -> QFTOptions:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> Output:
        """The logical output register."""

    def probability(self, outcome: str) -> float:
        """Return the ideal probability of an outcome."""

    def evaluate(self, counts: Mapping[str, int]) -> Evaluation:
        """Compare sampled counts with the ideal distribution."""

    def generate(self) -> mqt.core.mlir.QCProgram:
        """Generate the benchmark as a QC program."""

    @property
    def request_json(self) -> str:
        """The canonical request JSON."""

    @property
    def manifest_json(self) -> str:
        """The canonical manifest JSON."""

    @property
    def case_id(self) -> str:
        """The stable semantic case ID."""

    @staticmethod
    def from_request_json(json: str, *, source: str = "<request>") -> QFT:
        """Parse a strict benchmark request."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QFT:
        """Parse a strict benchmark manifest."""

class Phase:
    """An exact phase in turns modulo one turn."""

    def __init__(self, *, numerator: int, denominator: int) -> None: ...
    @property
    def numerator(self) -> int:
        """The reduced numerator."""

    @property
    def denominator(self) -> int:
        """The reduced denominator."""

class QPEMethod(enum.Enum):
    """Full-register or iterative measurement-and-feed-forward method."""

    STANDARD = 0

    ITERATIVE = 1

class QPEOptions:
    """Parameters for a QPE benchmark."""

    def __init__(self, *, precision: int, phase: fractions.Fraction | Phase, method: QPEMethod = ...) -> None: ...
    @property
    def precision(self) -> int:
        """The number of measured phase bits."""

    @property
    def phase(self) -> fractions.Fraction:
        """The reduced phase in turns."""

    @property
    def method(self) -> QPEMethod:
        """The circuit method."""

class QPE:
    """A validated QPE benchmark."""

    def __init__(self, options: QPEOptions) -> None: ...
    @property
    def options(self) -> QPEOptions:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> Output:
        """The logical output register."""

    def probability(self, outcome: str) -> float:
        """Return the ideal probability of an outcome."""

    def evaluate(self, counts: Mapping[str, int]) -> Evaluation:
        """Compare sampled counts with the ideal distribution."""

    def generate(self) -> mqt.core.mlir.QCProgram:
        """Generate the benchmark as a QC program."""

    @property
    def request_json(self) -> str:
        """The canonical request JSON."""

    @property
    def manifest_json(self) -> str:
        """The canonical manifest JSON."""

    @property
    def case_id(self) -> str:
        """The stable semantic case ID."""

    @staticmethod
    def from_request_json(json: str, *, source: str = "<request>") -> QPE:
        """Parse a strict benchmark request."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QPE:
        """Parse a strict benchmark manifest."""
