# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for proper site handling, zone filtering, and index remapping.

This test module was created in response to bugs where:
1. Zone sites were incorrectly counted as qubits (103 instead of 100)
2. Operation sites were misinterpreted (flat list instead of tuples of valid combinations)

These tests ensure that:
- Zone sites are properly filtered from qubit counts
- Site indices are remapped to logical qubit range [0, num_qubits)
- Operation sites follow QDMI specification structure
- Backend reports correct qubit count from device specification
"""

from __future__ import annotations

import importlib.util
from typing import cast

import pytest

from mqt.core import fomac

_qiskit_present = importlib.util.find_spec("qiskit") is not None

pytestmark = pytest.mark.skipif(not _qiskit_present, reason="qiskit not installed")

if _qiskit_present:
    from mqt.core.qdmi.qiskit import QiskitBackend


@pytest.fixture
def na_backend() -> QiskitBackend:
    """Fixture providing a QiskitBackend configured with the NA device.

    Returns:
        QiskitBackend instance configured with the MQT NA Default QDMI Device.

    Raises:
        RuntimeError: If the MQT NA Default QDMI Device is not found.

    Note:
        This fixture is used for tests that rely on specific NA device characteristics.
        These tests verify site handling, zone filtering, and QDMI specification compliance
        specific to the NA device structure (100 qubits + 3 zone sites).
    """
    devices_list = list(fomac.devices())
    for idx, device in enumerate(devices_list):
        if device.name() == "MQT NA Default QDMI Device":
            return QiskitBackend(device_index=idx)
    msg = "MQT NA Default QDMI Device not found"
    raise RuntimeError(msg)


@pytest.fixture
def na_device() -> fomac.Device:
    """Get the NA FoMaC device for direct inspection.

    Returns:
        The MQT NA Default QDMI Device.

    Raises:
        RuntimeError: If the MQT NA Default QDMI Device is not found.
    """
    devices_list = list(fomac.devices())
    for device in devices_list:
        if device.name() == "MQT NA Default QDMI Device":
            return device
    msg = "MQT NA Default QDMI Device not found"
    raise RuntimeError(msg)


class TestZoneSiteFiltering:
    """Tests for zone site filtering and exclusion from qubit counts."""

    def test_zone_sites_filtered_from_capabilities(self, na_backend: QiskitBackend, na_device: fomac.Device) -> None:  # noqa: PLR6301
        """Verify zone sites are not counted as qubits in capabilities."""
        # Get raw site data from device (materialize to avoid iterator exhaustion)
        all_sites = list(na_device.sites())
        # Treat only explicit True as a zone; anything else is non-zone
        zone_sites = [s for s in all_sites if s.is_zone() is True]
        non_zone_sites = [s for s in all_sites if s.is_zone() is not True]

        # Verify device has zone sites (our test device has 3)
        assert len(zone_sites) > 0, "Test device should have zone sites"

        # Verify capabilities only includes non-zone sites
        capabilities = na_backend._capabilities  # noqa: SLF001
        assert len(capabilities.sites) == len(non_zone_sites), (
            f"Capabilities should only include non-zone sites. "
            f"Expected {len(non_zone_sites)}, got {len(capabilities.sites)}"
        )

        # Verify no zone sites in capabilities by checking that all site indices
        # in capabilities are distinct and correspond to non-zone sites.
        # Note: QDMI does not mandate zero-based or consecutive indexing,
        # so we cannot assume indices are in range [0, num_qubits).
        capability_site_indices = {site_info.index for site_info in capabilities.sites}

        # All indices should be distinct (no duplicates)
        assert len(capability_site_indices) == len(capabilities.sites), (
            "Capability site indices should be distinct (no duplicates)"
        )

        # Number of capability sites should match number of non-zone sites
        assert len(capability_site_indices) == len(non_zone_sites), (
            f"Number of capability site indices ({len(capability_site_indices)}) should match "
            f"number of non-zone sites ({len(non_zone_sites)})"
        )

    def test_zone_sites_not_in_operation_sites(self, na_backend: QiskitBackend, na_device: fomac.Device) -> None:  # noqa: PLR6301
        """Verify operation sites only reference logical qubits, not zone sites.

        After remapping, operation sites should only contain indices in the range [0, num_qubits),
        which are logical qubit indices. Zone sites are filtered out during extraction.
        """
        num_qubits = na_backend._capabilities.num_qubits  # noqa: SLF001

        # Get the actual number of sites including zones from the device
        all_sites_count = len(list(na_device.sites()))

        # Verify that all_sites_count > num_qubits (i.e., there are zone sites)
        assert all_sites_count > num_qubits, (
            f"Test device should have zone sites. Total sites: {all_sites_count}, qubits: {num_qubits}"
        )

        # Check all operation sites in capabilities
        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            if op_info.sites is not None:
                # Flatten sites to check all indices
                if op_info.sites and isinstance(op_info.sites[0], tuple):
                    # Tuple of tuples - flatten to check all indices
                    sites_tuples = cast("tuple[tuple[int, ...], ...]", op_info.sites)
                    all_indices = {idx for site_tuple in sites_tuples for idx in site_tuple}
                else:
                    # Flat tuple
                    sites_flat = cast("tuple[int, ...]", op_info.sites)
                    all_indices = set(sites_flat)

                # Verify all indices are in valid logical qubit range [0, num_qubits)
                # This ensures zone sites (which had higher raw indices) are not present
                for idx in all_indices:
                    assert 0 <= idx < num_qubits, (
                        f"Operation '{op_name}' has index {idx} out of valid "
                        f"logical qubit range [0, {num_qubits}). "
                        f"Zone sites should not appear in remapped operation sites."
                    )

    def test_device_has_expected_zone_configuration(self, na_device: fomac.Device) -> None:  # noqa: PLR6301
        """Verify the NA device has the expected zone configuration."""
        # This test documents the expected structure of our NA test device
        all_sites = list(na_device.sites())
        zone_sites = [s for s in all_sites if s.is_zone()]

        # Our NA test device should have 3 zones:
        # - 1 for global single-qubit 'ry' operation
        # - 1 for global multi-qubit 'cz' operation
        # - 1 for shuttling unit
        assert len(zone_sites) == 3, (
            f"NA test device should have exactly 3 zone sites, found {len(zone_sites)}. "
            f"This test may need updating if device configuration changed."
        )

        # Verify zones have lower indices (they're created first in generator)
        zone_indices = sorted(s.index() for s in zone_sites)
        assert zone_indices == [0, 1, 2], f"Zone sites should have indices [0, 1, 2], got {zone_indices}"


class TestSiteIndexRemapping:
    """Tests for site index remapping to logical qubit range."""

    def test_site_indices_in_logical_range(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify all site indices in capabilities are in range [0, num_qubits)."""
        num_qubits = na_backend._capabilities.num_qubits  # noqa: SLF001

        # Check all site indices
        for site_info in na_backend._capabilities.sites:  # noqa: SLF001
            assert 0 <= site_info.index < num_qubits, (
                f"Site index {site_info.index} is out of logical qubit range [0, {num_qubits})"
            )

    def test_operation_site_indices_in_valid_range(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify all operation site indices are in valid logical qubit range."""
        num_qubits = na_backend._capabilities.num_qubits  # noqa: SLF001

        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            if op_info.sites is not None:
                # Check based on structure
                if op_info.sites and isinstance(op_info.sites[0], tuple):
                    # Tuple of tuples
                    sites_tuples = cast("tuple[tuple[int, ...], ...]", op_info.sites)
                    for site_tuple in sites_tuples:
                        for idx in site_tuple:
                            assert 0 <= idx < num_qubits, (
                                f"Operation '{op_name}' has site index {idx} out of valid range [0, {num_qubits})"
                            )
                else:
                    # Flat tuple
                    sites_flat = cast("tuple[int, ...]", op_info.sites)
                    for idx in sites_flat:
                        assert 0 <= idx < num_qubits, (
                            f"Operation '{op_name}' has site index {idx} out of valid range [0, {num_qubits})"
                        )

    def test_coupling_map_indices_in_valid_range(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify coupling map indices are in valid logical qubit range."""
        num_qubits = na_backend._capabilities.num_qubits  # noqa: SLF001
        coupling_map = na_backend._capabilities.coupling_map  # noqa: SLF001

        if coupling_map is not None:
            for idx0, idx1 in coupling_map:
                assert 0 <= idx0 < num_qubits, f"Coupling map has invalid index {idx0}, expected [0, {num_qubits})"
                assert 0 <= idx1 < num_qubits, f"Coupling map has invalid index {idx1}, expected [0, {num_qubits})"


class TestOperationSiteStructure:
    """Tests for operation site structure according to QDMI specification."""

    def test_local_multiqubit_operation_has_tuple_structure(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify local multi-qubit operations have tuple-of-tuples structure.

        According to QDMI specification:
        For local operations, the sites property returns a list of tuples.
        Each tuple contains sites and represents a valid combination for the operation.
        """
        # Find a local multi-qubit operation
        found_local_multiqubit = False

        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            # Local operations: not zoned, multi-qubit
            if not op_info.is_zoned and op_info.qubits_num is not None and op_info.qubits_num > 1:
                found_local_multiqubit = True

                # Must have sites defined
                assert op_info.sites is not None, f"Local multi-qubit operation '{op_name}' should have sites defined"

                # Must be non-empty
                assert len(op_info.sites) > 0, (
                    f"Local multi-qubit operation '{op_name}' should have at least one valid combination"
                )

                # First element must be a tuple
                assert isinstance(op_info.sites[0], tuple), (
                    f"Local multi-qubit operation '{op_name}' should have tuple-of-tuples structure, "
                    f"but first element is {type(op_info.sites[0])}"
                )

                # Each tuple should have qubits_num elements
                for site_tuple in op_info.sites:
                    assert isinstance(site_tuple, tuple), f"Each site combination in '{op_name}' should be a tuple"
                    assert len(site_tuple) == op_info.qubits_num, (
                        f"Each site combination in '{op_name}' should have {op_info.qubits_num} qubits, "
                        f"found {len(site_tuple)}"
                    )

        assert found_local_multiqubit, "NA test device should have at least one local multi-qubit operation to test"

    def test_single_qubit_operation_has_flat_structure(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify single-qubit operations have flat tuple structure."""
        # Find single-qubit operations (local or global)
        found_single_qubit = False

        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            if op_info.qubits_num == 1 and op_info.sites is not None:
                found_single_qubit = True

                # Must be non-empty
                assert len(op_info.sites) > 0, f"Single-qubit operation '{op_name}' should have at least one site"

                # First element should be an int, not a tuple
                assert isinstance(op_info.sites[0], int), (
                    f"Single-qubit operation '{op_name}' should have flat tuple structure, "
                    f"but first element is {type(op_info.sites[0])}"
                )

        assert found_single_qubit, "NA test device should have at least one single-qubit operation to test"

    def test_global_operations_identified_correctly(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify global (zoned) operations are properly identified."""
        # Find global operations
        global_ops = {
            op_name: op_info
            for op_name, op_info in na_backend._capabilities.operations.items()  # noqa: SLF001
            if op_info.is_zoned
        }

        # Our NA test device should have global 'ry' (single-qubit, global)
        # and possibly global 'cz' (multi-qubit, global)
        assert len(global_ops) > 0, "NA test device should have at least one global operation"

        # Verify global operations have is_zoned = True
        for op_name, op_info in global_ops.items():
            assert op_info.is_zoned is True, f"Global operation '{op_name}' should have is_zoned=True"

    def test_local_cz_operation_structure(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify the local 'cz' operation has correct structure.

        This is a regression test for the specific bug where local 'cz' operation
        was returning a flat list of 1004 elements instead of 502 tuples of pairs.
        """
        # Find local cz operation
        local_cz = None
        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            if op_name.lower() == "cz" and not op_info.is_zoned:
                local_cz = op_info
                break

        if local_cz is None:
            pytest.skip("NA test device doesn't have a local 'cz' operation")

        # Verify structure
        assert local_cz.sites is not None, "Local 'cz' should have sites defined"
        assert len(local_cz.sites) > 0, "Local 'cz' should have valid combinations"
        assert isinstance(local_cz.sites[0], tuple), "Local 'cz' should have tuple-of-tuples structure"
        assert len(local_cz.sites[0]) == 2, "Each 'cz' combination should have 2 qubits"

        # Verify we have the expected number of valid pairs
        # (not all possible pairs, but device-specific valid combinations)
        num_pairs = len(local_cz.sites)
        num_qubits = na_backend._capabilities.num_qubits  # noqa: SLF001
        max_possible_pairs = num_qubits * (num_qubits - 1) // 2

        assert num_pairs > 0, "Should have at least one valid 'cz' pair"
        assert num_pairs <= max_possible_pairs, (
            f"Number of valid 'cz' pairs ({num_pairs}) should not exceed maximum possible pairs ({max_possible_pairs})"
        )


class TestBackendQubitCount:
    """Tests for backend qubit count correctness."""

    def test_backend_num_qubits_matches_device_spec(self, na_backend: QiskitBackend, na_device: fomac.Device) -> None:  # noqa: PLR6301
        """Verify backend reports correct qubit count from device specification.

        This is a regression test for the bug where backend.target.num_qubits
        reported 103 instead of 100 due to zone sites being counted.
        """
        device_qubits = na_device.qubits_num()
        backend_qubits = na_backend.target.num_qubits

        assert backend_qubits == device_qubits, (
            f"Backend should report {device_qubits} qubits (from device spec), but reports {backend_qubits}"
        )

    def test_target_num_qubits_equals_capabilities_num_qubits(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify backend.target.num_qubits matches capabilities.num_qubits."""
        capabilities_qubits = na_backend._capabilities.num_qubits  # noqa: SLF001
        target_qubits = na_backend.target.num_qubits

        assert target_qubits == capabilities_qubits, (
            f"Target num_qubits ({target_qubits}) should match capabilities num_qubits ({capabilities_qubits})"
        )

    def test_num_qubits_matches_site_count(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify number of qubits matches number of sites in capabilities."""
        num_qubits = na_backend._capabilities.num_qubits  # noqa: SLF001
        num_sites = len(na_backend._capabilities.sites)  # noqa: SLF001

        assert num_qubits == num_sites, (
            f"Number of qubits ({num_qubits}) should match number of sites ({num_sites}) in capabilities"
        )

    def test_device_json_specifies_100_qubits(self, na_device: fomac.Device) -> None:  # noqa: PLR6301
        """Verify the NA device JSON specifies 100 qubits.

        This test documents the expected configuration of our NA test device
        and will fail if the device JSON is changed.
        """
        expected_qubits = 100
        actual_qubits = na_device.qubits_num()

        assert actual_qubits == expected_qubits, (
            f"NA test device should have {expected_qubits} qubits "
            f"(as specified in json/na/device.json), found {actual_qubits}"
        )


class TestQDMISpecificationCompliance:
    """Tests for QDMI specification compliance."""

    def test_operation_sites_follow_qdmi_spec(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify operation sites follow QDMI specification.

        QDMI Specification for QDMI_OPERATION_PROPERTY_SITES:
        - For local operations: returns a list of tuples where each tuple
          contains sites representing a valid combination for the operation.
        - For global operations: returns a list of zone sites where the
          operation can be applied.
        """
        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            if op_info.sites is None:
                # Operation applies to all qubits - this is valid
                continue

            if not op_info.is_zoned and op_info.qubits_num and op_info.qubits_num > 1:
                # Local multi-qubit operation: should have tuple-of-tuples
                assert isinstance(op_info.sites[0], tuple), (
                    f"Local multi-qubit operation '{op_name}' should have tuple-of-tuples structure "
                    f"per QDMI specification"
                )

                # Each tuple should have qubits_num elements
                sites_tuples = cast("tuple[tuple[int, ...], ...]", op_info.sites)
                for site_tuple in sites_tuples:
                    assert len(site_tuple) == op_info.qubits_num, (
                        f"Each site combination in '{op_name}' should have {op_info.qubits_num} elements "
                        f"per QDMI specification"
                    )

            elif op_info.qubits_num == 1:
                # Single-qubit operation: should have flat tuple of individual sites
                assert isinstance(op_info.sites[0], int), (
                    f"Single-qubit operation '{op_name}' should have flat tuple structure per QDMI specification"
                )

    def test_sites_property_type_consistency(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """Verify sites property has consistent type throughout each operation."""
        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            if op_info.sites is None or len(op_info.sites) == 0:
                continue

            # Check type of first element
            first_elem_type = type(op_info.sites[0])

            # All elements should have the same type
            for i, elem in enumerate(op_info.sites):
                assert isinstance(elem, first_elem_type), (  # Use isinstance instead of type comparison
                    f"Operation '{op_name}' has inconsistent site types: "
                    f"element 0 is {first_elem_type}, element {i} is {type(elem)}"
                )


class TestRegressionPrevention:
    """Tests that explicitly check for the original bugs to prevent regression."""

    def test_regression_zone_sites_not_counted_as_qubits(  # noqa: PLR6301
        self, na_backend: QiskitBackend, na_device: fomac.Device
    ) -> None:
        """REGRESSION TEST: Verify zone sites are not counted as qubits.

        Original Bug: Backend reported 103 qubits instead of 100 because
        zone sites (3) were being counted along with regular qubit sites (100).
        """
        # Get zone site count
        zone_count = len([s for s in na_device.sites() if s.is_zone()])

        # Get device spec qubit count
        device_qubits = na_device.qubits_num()

        # Get backend qubit count
        backend_qubits = na_backend.target.num_qubits

        # Backend should NOT equal device_qubits + zone_count
        assert backend_qubits != device_qubits + zone_count, (
            f"REGRESSION: Backend is counting zone sites as qubits! "
            f"Backend reports {backend_qubits} qubits = {device_qubits} device qubits + {zone_count} zones. "
            f"Zone sites should be filtered out."
        )

        # Backend should equal device_qubits (without zones)
        assert backend_qubits == device_qubits, (
            f"Backend should report {device_qubits} qubits (from device spec), but reports {backend_qubits}"
        )

    def test_regression_local_multiqubit_ops_not_flat_list(self, na_backend: QiskitBackend) -> None:  # noqa: PLR6301
        """REGRESSION TEST: Verify local multi-qubit operations don't return flat list.

        Original Bug: Local 'cz' operation returned a flat list of 1004 site indices
        instead of a tuple of 502 pairs. This caused incorrect generation of all possible
        pairs instead of using device-specific valid combinations.
        """
        # Find local multi-qubit operations
        for op_name, op_info in na_backend._capabilities.operations.items():  # noqa: SLF001
            if not op_info.is_zoned and op_info.qubits_num and op_info.qubits_num > 1:
                if op_info.sites is None:
                    continue

                # Should NOT be a flat list of integers
                assert not isinstance(op_info.sites[0], int), (
                    f"REGRESSION: Local multi-qubit operation '{op_name}' has flat list structure! "
                    f"Should have tuple-of-tuples structure per QDMI specification. "
                    f"First element type: {type(op_info.sites[0])}"
                )

                # Should be tuple of tuples
                assert isinstance(op_info.sites[0], tuple), (
                    f"Local multi-qubit operation '{op_name}' should have tuple-of-tuples structure"
                )

                # If qubits_num is 2, each inner tuple should have 2 elements
                if op_info.qubits_num == 2:
                    sites_tuples = cast("tuple[tuple[int, ...], ...]", op_info.sites)
                    for site_tuple in sites_tuples:
                        assert len(site_tuple) == 2, (
                            f"REGRESSION: Local 2-qubit operation '{op_name}' has tuple with "
                            f"{len(site_tuple)} elements instead of 2. "
                            f"Site structure may be incorrectly parsed."
                        )
