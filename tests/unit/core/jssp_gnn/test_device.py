"""
Unit tests for device selection utilities.
"""

import pytest
import torch

from jssp_gnn.utils.device import get_device, set_device


@pytest.mark.unit
class TestDeviceUtils:
    """Test device configuration and retrieval."""

    def test_get_set_device(self):
        """Test basic get/set functionality."""
        original_device = get_device()

        try:
            # Set to CPU explicitly
            set_device("cpu")
            assert get_device().type == "cpu"

            # Set using torch.device object
            new_device = torch.device("cpu")
            set_device(new_device)
            assert get_device() == new_device

        finally:
            # Restore original device to avoid side effects on other tests
            set_device(original_device)

    def test_set_device_auto(self):
        """Test 'auto' device selection."""
        original_device = get_device()

        try:
            set_device("auto")
            expected_type = "cuda" if torch.cuda.is_available() else "cpu"
            assert get_device().type == expected_type
        finally:
            set_device(original_device)
