"""TPU Tensor Operations for CayleyPy.

This module provides TPU-accelerated tensor operations with native int64 support,
optimized for TPU v6e (Trillium) architecture with 256x256 systolic arrays.
"""

import logging
from typing import Tuple, Dict, Any, Optional

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    nnx = None  # type: ignore

from .tpu_backend import TPUBackend


class TPUTensorOpsModule(nnx.Module):
    """NNX module for TPU-accelerated tensor operations with native int64 support."""

    def __init__(self, backend: TPUBackend, rngs: Optional[nnx.Rngs] = None):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU tensor operations")

        self.backend = backend

        # Cache for frequently computed operations
        self.operation_cache = nnx.Variable({})

        # Performance metrics
        self.metrics = nnx.Variable(
            {
                "operations_count": 0,
                "cache_hits": 0,
                "total_elements_processed": 0,
                "int64_operations": 0,
                "systolic_array_utilization": 0.0,
                "memory_peak_mb": 0.0,
                "unique_operations": 0,
                "isin_operations": 0,
                "permutation_operations": 0,
                "deduplication_operations": 0,
            }
        )

        self.logger = logging.getL
