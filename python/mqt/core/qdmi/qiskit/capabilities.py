# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Device capability extraction & caching.

This module provides lightweight, dependency-free (no Qiskit required) data
classes and helper functions to introspect a FoMaC-backed QDMI device and
produce a normalized, hashable capability snapshot. The snapshot is suitable
for embedding in higher-level backend metadata (e.g., Qiskit BackendV2 Target).

Design goals:
- Pure Python; safe to import regardless of optional extras.
- Avoid repeated expensive native calls via an object cache keyed by a
  deterministic *device signature* (structural hash of salient properties).
- Preserve optional values (None) without interpretation; semantic derivation
  is deferred to later layers.
- Provide stable dataclasses forming part of an intended public extension API
  (users may reference them for plugin development or inspection).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.core import fomac

__all__ = [
    "DeviceCapabilities",
    "DeviceOperationInfo",
    "DeviceSiteInfo",
    "extract_capabilities",
    "get_capabilities",
]

# ---------------------------------------------------------------------------
# Dataclasses representing a normalized capability snapshot
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeviceSiteInfo:
    """Normalized per-site (potential qubit location) metadata.

    All attributes retain their native units / types; no scaling or
    interpretation beyond direct extraction is applied at this layer.
    """

    index: int
    name: str | None
    x: int | None
    y: int | None
    z: int | None
    is_zone: bool | None
    x_extent: int | None
    y_extent: int | None
    z_extent: int | None
    module_index: int | None
    submodule_index: int | None
    t1: int | None
    t2: int | None


@dataclass(slots=True)
class DeviceOperationInfo:
    """Normalized per-operation metadata.

    The ``sites`` field contains a tuple of site indices if the operation
    enumerates specific allowed sites; ``None`` otherwise.
    """

    name: str
    qubits_num: int | None
    parameters_num: int
    duration: int | None
    fidelity: float | None
    interaction_radius: int | None
    blocking_radius: int | None
    idling_fidelity: float | None
    is_zoned: bool | None
    mean_shuttling_speed: int | None
    sites: tuple[int, ...] | None


@dataclass(slots=True)
class DeviceCapabilities:
    """Aggregate capability snapshot for a FoMaC device.

    The ``signature`` captures stable structural attributes (name, version,
    number of qubits, operation name set, coupling map) to enable cache reuse
    across minor performance fluctuations. ``capabilities_hash`` is a SHA256 of
    a canonical JSON representation (including optional properties) and thus
    changes whenever *any* included field changes.
    """

    device_name: str
    device_version: str
    library_version: str
    num_qubits: int
    duration_unit: str | None
    duration_scale_factor: float | None
    length_unit: str | None
    length_scale_factor: float | None
    min_atom_distance: int | None
    status: str | None
    sites: list[DeviceSiteInfo] = field(default_factory=list)
    operations: dict[str, DeviceOperationInfo] = field(default_factory=dict)
    coupling_map: list[tuple[int, int]] | None = None
    signature: str | None = None
    capabilities_hash: str | None = None

    def to_canonical_json(self) -> str:
        """Return a deterministic JSON string representation.

        Excludes the ``capabilities_hash`` field itself to avoid recursion.

        Returns:
            Canonical JSON string with sorted keys and compact separators.
        """
        obj = {
            "device_name": self.device_name,
            "device_version": self.device_version,
            "library_version": self.library_version,
            "num_qubits": self.num_qubits,
            "duration_unit": self.duration_unit,
            "duration_scale_factor": self.duration_scale_factor,
            "length_unit": self.length_unit,
            "length_scale_factor": self.length_scale_factor,
            "min_atom_distance": self.min_atom_distance,
            "status": self.status,
            "sites": [asdict(s) for s in self.sites],
            "operations": {name: asdict(op) for name, op in sorted(self.operations.items())},
            "coupling_map": self.coupling_map,
            "signature": self.signature,
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))


_capability_cache: dict[str, DeviceCapabilities] = {}
_cache_lock = threading.Lock()


def _compute_device_signature(device: fomac.Device) -> str:
    """Compute a concise structural signature string for a device.

    The signature *excludes* performance / temporal metrics that may fluctuate
    (e.g., site coherence times) to maximize cache hit utility. It captures the
    static structure relevant for backend Target construction.

    Returns:
        JSON string (compact, sorted keys) capturing structural attributes.
    """
    name = device.name()
    version = device.version()
    num_qubits = device.qubits_num()
    op_names = sorted({op.name() for op in device.operations()})
    coupling = device.coupling_map()
    coupling_indices = sorted((c[0].index(), c[1].index()) for c in coupling) if coupling is not None else None
    signature_obj = {
        "name": name,
        "version": version,
        "num_qubits": num_qubits,
        "operations": op_names,
        "coupling": coupling_indices,
    }
    return json.dumps(signature_obj, sort_keys=True, separators=(",", ":"))


def _safe_site_info(s: fomac.Device.Site) -> DeviceSiteInfo | None:
    """Safely build a DeviceSiteInfo for a site or return None on failure.

    Returns:
        A populated DeviceSiteInfo instance, or ``None`` if extraction failed.
    """
    try:
        return DeviceSiteInfo(
            index=s.index(),
            name=s.name(),
            x=s.x_coordinate(),
            y=s.y_coordinate(),
            z=s.z_coordinate(),
            is_zone=s.is_zone(),
            x_extent=s.x_extent(),
            y_extent=s.y_extent(),
            z_extent=s.z_extent(),
            module_index=s.module_index(),
            submodule_index=s.submodule_index(),
            t1=s.t1(),
            t2=s.t2(),
        )
    except Exception:  # noqa: BLE001
        return None


def _safe_operation_info(op: fomac.Device.Operation) -> DeviceOperationInfo | None:
    """Safely build a DeviceOperationInfo for an operation or return None.

    Returns:
        A populated DeviceOperationInfo instance, or ``None`` if extraction failed.
    """
    name = op.name()
    site_list = op.sites()
    site_indices = tuple(sorted(s.index() for s in site_list)) if site_list is not None else None
    return DeviceOperationInfo(
        name=name,
        qubits_num=getattr(op, "qubits_num", lambda: None)(),
        parameters_num=getattr(op, "parameters_num", lambda: 0)(),
        duration=getattr(op, "duration", lambda: None)(),
        fidelity=getattr(op, "fidelity", lambda: None)(),
        interaction_radius=getattr(op, "interaction_radius", lambda: None)(),
        blocking_radius=getattr(op, "blocking_radius", lambda: None)(),
        idling_fidelity=getattr(op, "idling_fidelity", lambda: None)(),
        is_zoned=getattr(op, "is_zoned", lambda: None)(),
        mean_shuttling_speed=getattr(op, "mean_shuttling_speed", lambda: None)(),
        sites=site_indices,
    )


def extract_capabilities(device: fomac.Device) -> DeviceCapabilities:
    """Extract full capabilities from a FoMaC device (no caching).

    Returns:
        Fresh :class:`DeviceCapabilities` snapshot.
    """
    signature = _compute_device_signature(device)

    site_infos: list[DeviceSiteInfo] = []
    for s in sorted(device.sites(), key=lambda x: x.index()):
        site_info = _safe_site_info(s)
        if site_info is not None:
            site_infos.append(site_info)

    op_infos: dict[str, DeviceOperationInfo] = {}
    for op in device.operations():
        op_info = _safe_operation_info(op)
        if op_info is not None:
            op_infos[op_info.name] = op_info

    cm = device.coupling_map()
    coupling_indices_list = sorted((pair[0].index(), pair[1].index()) for pair in cm) if cm is not None else None

    status = device.status()
    cap = DeviceCapabilities(
        device_name=device.name(),
        device_version=device.version(),
        library_version=device.library_version(),
        num_qubits=device.qubits_num(),
        duration_unit=device.duration_unit(),
        duration_scale_factor=device.duration_scale_factor(),
        length_unit=device.length_unit(),
        length_scale_factor=device.length_scale_factor(),
        min_atom_distance=device.min_atom_distance(),
        status=getattr(status, "name", None) or str(status),
        sites=site_infos,
        operations=op_infos,
        coupling_map=coupling_indices_list,
        signature=signature,
    )
    cap.capabilities_hash = sha256(cap.to_canonical_json().encode("utf-8")).hexdigest()
    return cap


def get_capabilities(device: fomac.Device, *, use_cache: bool = True) -> DeviceCapabilities:
    """Return cached capabilities for ``device`` (extracting if necessary).

    Args:
        device: FoMaC device instance.
        use_cache: If ``False``, force re-extraction (cache not updated).

    Returns:
        Cached or freshly extracted :class:`DeviceCapabilities` snapshot.
    """
    signature = _compute_device_signature(device)
    if use_cache:
        with _cache_lock:
            cached = _capability_cache.get(signature)
            if cached is not None:
                return cached
    cap = extract_capabilities(device)
    if use_cache:
        with _cache_lock:
            _capability_cache[signature] = cap
    return cap


def _cache_info() -> dict[str, int]:  # pragma: no cover - debug helper
    """Return basic cache metrics (for diagnostics)."""
    return {"entries": len(_capability_cache)}
