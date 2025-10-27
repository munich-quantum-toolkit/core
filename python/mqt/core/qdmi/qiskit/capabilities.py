# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Device capability extraction.

This module provides lightweight, dependency-free (no Qiskit required) data
classes and helper functions to introspect a FoMaC-backed QDMI device and
produce a normalized, hashable capability snapshot. The snapshot is suitable
for embedding in higher-level backend metadata (e.g., Qiskit BackendV2 Target).

Design goals:
- Pure Python; safe to import regardless of optional extras.
- Preserve optional values (None) without interpretation; semantic derivation
  is deferred to later layers.
- Provide stable dataclasses forming part of an intended public extension API
  (users may reference them for plugin development or inspection).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mqt.core import fomac

__all__ = [
    "DeviceCapabilities",
    "DeviceOperationInfo",
    "DeviceSiteInfo",
    "extract_capabilities",
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

    The ``sites`` field interpretation depends on the operation type:
    - For local multi-qubit operations: tuple of tuples, where each inner tuple
        represents a valid combination of site indices for the operation.
    - For single-qubit operations: tuple of individual site indices.
    - For global (zoned) operations: tuple of zone site indices.
    - ``None`` if the operation applies to all qubits.
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
    sites: tuple[int, ...] | tuple[tuple[int, ...], ...] | None


@dataclass(slots=True)
class DeviceCapabilities:
    """Aggregate capability snapshot for a FoMaC device.

    The ``signature`` captures stable structural attributes (name, version,
    number of qubits, operation name set, coupling map). ``capabilities_hash``
    is a SHA256 of a canonical JSON representation (including optional properties)
    and thus changes whenever *any* included field changes.
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


def _compute_device_signature(device: fomac.Device) -> str:
    """Compute a concise structural signature string for a device.

    The signature captures the static structure relevant for backend Target construction.

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


def _build_site_info(s: fomac.Device.Site) -> DeviceSiteInfo:
    """Build a DeviceSiteInfo for a site.

    Args:
        s: Site from FoMaC device.

    Returns:
        A populated DeviceSiteInfo instance.

    Raises:
        Any exception from accessing site properties, indicating a device specification issue.
    """
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


def _build_operation_info(op: fomac.Device.Operation) -> DeviceOperationInfo:
    """Build a DeviceOperationInfo for an operation.

    Interprets the sites list according to QDMI specification:
    - For local multi-qubit operations: returns tuples of site combinations
    - For single-qubit and global operations: returns individual sites

    Args:
        op: Operation from FoMaC device.

    Returns:
        A populated DeviceOperationInfo instance.

    Raises:
        Any exception from accessing operation properties, indicating a device specification issue.
    """
    name = op.name()
    site_list = op.sites()

    # Determine how to interpret the sites based on operation type
    qubits_num = op.qubits_num()
    is_zoned = op.is_zoned()

    site_indices: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None

    if site_list is not None:
        # Extract raw site indices
        raw_indices = [s.index() for s in site_list]

        # For local multi-qubit operations, the flat list represents consecutive pairs
        # due to reinterpret_cast from vector<pair<Site, Site>> to vector<Site>
        if not is_zoned and qubits_num is not None and qubits_num > 1:
            # Group consecutive elements into tuples of size qubits_num
            if len(raw_indices) % qubits_num == 0:
                site_tuples = [tuple(raw_indices[i : i + qubits_num]) for i in range(0, len(raw_indices), qubits_num)]
                site_indices = tuple(site_tuples)
            else:
                # Fallback: treat as flat list if not evenly divisible
                site_indices = tuple(sorted(set(raw_indices)))
        else:
            # For single-qubit and global operations, use flat list
            site_indices = tuple(sorted(set(raw_indices)))

    return DeviceOperationInfo(
        name=name,
        qubits_num=qubits_num,
        parameters_num=op.parameters_num(),
        duration=op.duration(),
        fidelity=op.fidelity(),
        interaction_radius=op.interaction_radius(),
        blocking_radius=op.blocking_radius(),
        idling_fidelity=op.idling_fidelity(),
        is_zoned=is_zoned,
        mean_shuttling_speed=op.mean_shuttling_speed(),
        sites=site_indices,
    )


def _remap_operation_sites(
    op_info: DeviceOperationInfo,
    site_index_to_qubit_index: dict[int, int],
) -> DeviceOperationInfo:
    """Remap site indices in an operation to logical qubit indices.

    Args:
        op_info: Operation info with raw site indices.
        site_index_to_qubit_index: Mapping from raw site indices to logical qubit indices.

    Returns:
        Operation info with remapped site indices.
    """
    if op_info.sites is None:
        return op_info

    # Check if sites is a tuple of tuples (local multi-qubit) or flat tuple
    if op_info.sites and isinstance(op_info.sites[0], tuple):
        # Local multi-qubit operation: remap each tuple
        # Cast to help mypy understand we're in the tuple[tuple[int, ...], ...] branch
        sites_as_tuples = cast("tuple[tuple[int, ...], ...]", op_info.sites)
        remapped_tuples = []
        for site_tuple in sites_as_tuples:
            remapped_tuple = tuple(
                site_index_to_qubit_index[site_idx] for site_idx in site_tuple if site_idx in site_index_to_qubit_index
            )
            if len(remapped_tuple) == len(site_tuple):  # Only include if all sites mapped
                remapped_tuples.append(remapped_tuple)
        op_info.sites = tuple(remapped_tuples) if remapped_tuples else None
    else:
        # Single-qubit or global operation: remap flat list
        # Cast to help mypy understand we're in the tuple[int, ...] branch
        sites_as_ints = cast("tuple[int, ...]", op_info.sites)
        remapped_sites = [
            site_index_to_qubit_index[site_idx] for site_idx in sites_as_ints if site_idx in site_index_to_qubit_index
        ]
        op_info.sites = tuple(sorted(remapped_sites)) if remapped_sites else None

    return op_info


def extract_capabilities(device: fomac.Device) -> DeviceCapabilities:
    """Extract full capabilities from a FoMaC device.

    Returns:
        Fresh :class:`DeviceCapabilities` snapshot.
    """
    signature = _compute_device_signature(device)

    # Filter out zone sites - they are not qubits
    all_sites = sorted(device.sites(), key=lambda x: x.index())
    non_zone_sites = [s for s in all_sites if not s.is_zone()]

    # Create a mapping from site index to logical qubit index (0-based, excluding zones)
    site_index_to_qubit_index = {s.index(): i for i, s in enumerate(non_zone_sites)}

    site_infos: list[DeviceSiteInfo] = []
    for s in non_zone_sites:
        site_info = _build_site_info(s)
        # Remap the index to the logical qubit index
        site_info.index = site_index_to_qubit_index[s.index()]
        site_infos.append(site_info)

    op_infos: dict[str, DeviceOperationInfo] = {}
    for op in device.operations():
        op_info = _build_operation_info(op)
        # Remap site indices in operations to logical qubit indices
        op_info = _remap_operation_sites(op_info, site_index_to_qubit_index)
        op_infos[op_info.name] = op_info

    cm = device.coupling_map()
    coupling_indices_list = None
    if cm is not None:
        # Remap coupling map indices to logical qubit indices
        remapped_coupling = []
        for pair in cm:
            idx0, idx1 = pair[0].index(), pair[1].index()
            if idx0 in site_index_to_qubit_index and idx1 in site_index_to_qubit_index:
                remapped_coupling.append((site_index_to_qubit_index[idx0], site_index_to_qubit_index[idx1]))
        coupling_indices_list = sorted(remapped_coupling) if remapped_coupling else None

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
        status=status.name,
        sites=site_infos,
        operations=op_infos,
        coupling_map=coupling_indices_list,
        signature=signature,
    )
    cap.capabilities_hash = sha256(cap.to_canonical_json().encode("utf-8")).hexdigest()
    return cap
