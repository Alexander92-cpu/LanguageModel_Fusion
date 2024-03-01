"""
In this script we check that all basic training processes are working in this project.
"""

import pytest
import torch


class TestCUDA:
    """
    Check if CUDA works
    """
    @pytest.mark.run(order=1)
    def test_cuda_availability(self):
        """
        Check if CUDA is available
        """
        assert torch.cuda.is_available(), "CUDA is not available."

    @pytest.mark.run(order=2)
    def test_cuda_nums(self):
        """
        Check if number of available CUDAs is positive
        """
        assert torch.cuda.device_count() > 0, "No CUDA devices found."
