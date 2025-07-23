"""JAX implementation of StringEncoder for efficient state encoding/decoding.

This module provides a JAX-based implementation of the StringEncoder class,
optimized for TPU/GPU computation with JIT compilation and vectorization.
"""

import math
from typing import Callable, Dict, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore

# We are using int64, but avoid using the sign bit.
CODEWORD_LENGTH = 63


def _check_jax_available():
    """Check if JAX is available and raise error if not."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not available. Install with: pip install 'cayleypy[jax-cpu]', "
            "'cayleypy[jax-cuda]', or 'cayleypy[jax-tpu]'"
        )


class JAXStringEncoder:
    """JAX-based helper class to encode strings that represent elements of coset.

    Original (decoded) strings are 2D arrays where array elements are integers representing elements being permuted.
    In encoded format, these elements are compressed to take less memory. Each element takes only `code_width` bits.
    For binary strings (`code_width=1`) and `n<=63`, this allows to represent coset element with a single int64 number.
    Elements in the original string must be in range `[0, 2**code_width)`.
    This class also provides functionality to efficiently apply permutation in encoded format using bit operations.
    """

    def __init__(self, *, code_width: int = 1, n: int = 1):
        """Initializes JAXStringEncoder.

        :param code_width: Number of bits to encode one element of coset.
        :param n: Length of the string to encode.
        """
        _check_jax_available()

        assert 1 <= code_width <= CODEWORD_LENGTH
        self.w = code_width
        self.n = n
        self.encoded_length = int(math.ceil(self.n * self.w / CODEWORD_LENGTH))  # Encoded length.

    def encode(self, s: jnp.ndarray) -> jnp.ndarray:
        """Encodes array of coset elements.

        Input shape `(m, self.n)`. Output shape `(m, self.encoded_length)`.
        """
        assert len(s.shape) == 2
        assert s.shape[1] == self.n
        assert jnp.min(s) >= 0, "Cannot encode negative values."
        max_value = jnp.max(s)
        assert max_value < 2**self.w, f"Width {self.w} is not sufficient to encode value {max_value}."

        encoded = jnp.zeros((s.shape[0], self.encoded_length), dtype=jnp.int64)
        w, cl = self.w, CODEWORD_LENGTH

        # Match PyTorch implementation exactly
        for i in range(w * self.n):
            element_idx = i // w
            bit_in_element = i % w
            codeword_idx = i // cl
            bit_in_codeword = i % cl

            # Extract the specific bit from the element
            bit_value = (s[:, element_idx] >> bit_in_element) & 1
            # Set the bit in the encoded array
            encoded = encoded.at[:, codeword_idx].set(encoded[:, codeword_idx] | (bit_value << bit_in_codeword))

        return encoded

    def decode(self, encoded: jnp.ndarray) -> jnp.ndarray:
        """Decodes array of coset elements.

        Input shape `(m, self.encoded_length)`. Output shape `(m, self.n)`.
        """
        orig = jnp.zeros((encoded.shape[0], self.n), dtype=jnp.int64)
        w, cl = self.w, CODEWORD_LENGTH

        # Match PyTorch implementation exactly
        for i in range(w * self.n):
            element_idx = i // w
            bit_in_element = i % w
            codeword_idx = i // cl
            bit_in_codeword = i % cl

            # Extract the specific bit from the encoded array
            bit_value = (encoded[:, codeword_idx] >> bit_in_codeword) & 1
            # Set the bit in the original array
            orig = orig.at[:, element_idx].set(orig[:, element_idx] | (bit_value << bit_in_element))

        return orig

    def prepare_shift_to_mask(self, p: list[int]) -> Dict[Tuple[int, int, int], int]:
        """Prepare shift-to-mask mapping for permutation implementation.

        Args:
            p: Permutation as list of integers

        Returns:
            Dictionary mapping (start_cw_id, end_cw_id, shift) to mask
        """
        assert len(p) == self.n
        shift_to_mask: Dict[Tuple[int, int, int], int] = {}

        for i in range(self.n):
            for j in range(self.w):
                start_bit = int(p[i] * self.w + j)
                end_bit = i * self.w + j
                start_cw_id = start_bit // CODEWORD_LENGTH
                end_cw_id = end_bit // CODEWORD_LENGTH
                shift = (end_bit % CODEWORD_LENGTH) - (start_bit % CODEWORD_LENGTH)
                key = (start_cw_id, end_cw_id, shift)
                if key not in shift_to_mask:
                    shift_to_mask[key] = 0
                shift_to_mask[key] |= 1 << (start_bit % CODEWORD_LENGTH)

        return shift_to_mask

    def implement_permutation(self, p: list[int]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Converts permutation to a JIT-compiled function on encoded array implementing this permutation.

        Args:
            p: Permutation as list of integers

        Returns:
            JIT-compiled function that applies permutation to encoded arrays
        """
        shift_to_mask = self.prepare_shift_to_mask(p)

        @jit
        def permutation_func(x: jnp.ndarray) -> jnp.ndarray:
            """Apply permutation to encoded array."""
            y = jnp.zeros_like(x)

            for (start_cw_id, end_cw_id, shift), mask in shift_to_mask.items():
                if shift > 0:
                    y = y.at[:, end_cw_id].set(y[:, end_cw_id] | ((x[:, start_cw_id] & mask) << shift))
                elif shift < 0:
                    y = y.at[:, end_cw_id].set(y[:, end_cw_id] | ((x[:, start_cw_id] & mask) >> (-shift)))
                else:
                    y = y.at[:, end_cw_id].set(y[:, end_cw_id] | (x[:, start_cw_id] & mask))

            return y

        return permutation_func

    def implement_permutation_1d(self, p: list[int]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Converts permutation to a function on encoded array implementing this permutation.

        The function converts 1D array to 1D array of the same dimension.
        Applicable only if state can be encoded by single int64 (encoded_length=1).

        Args:
            p: Permutation as list of integers

        Returns:
            JIT-compiled function for 1D permutation
        """
        assert self.encoded_length == 1, "1D permutation only applicable when encoded_length=1"

        shift_to_mask = self.prepare_shift_to_mask(p)

        @jit
        def permutation_1d_func(x: jnp.ndarray) -> jnp.ndarray:
            """Apply permutation to 1D encoded array."""
            result = jnp.zeros_like(x)

            for (_, _, shift), mask in shift_to_mask.items():
                if shift > 0:
                    result = result | ((x & mask) << shift)
                elif shift < 0:
                    result = result | ((x & mask) >> (-shift))
                else:
                    result = result | (x & mask)

            return result

        return permutation_1d_func

    # Vectorized operations using vmap
    def encode_batch(self, s: jnp.ndarray) -> jnp.ndarray:
        """Vectorized batch encoding using vmap.

        Args:
            s: Batch of states to encode, shape (batch_size, n)

        Returns:
            Encoded states, shape (batch_size, encoded_length)
        """
        return self.encode(s)

    def decode_batch(self, encoded: jnp.ndarray) -> jnp.ndarray:
        """Vectorized batch decoding using vmap.

        Args:
            encoded: Batch of encoded states, shape (batch_size, encoded_length)

        Returns:
            Decoded states, shape (batch_size, n)
        """
        return self.decode(encoded)

    def apply_permutation_to_batch(self, encoded_states: jnp.ndarray, permutation: list[int]) -> jnp.ndarray:
        """Apply permutation to a batch of encoded states.

        Args:
            encoded_states: Batch of encoded states
            permutation: Permutation to apply

        Returns:
            Batch of permuted encoded states
        """
        perm_func = self.implement_permutation(permutation)
        return perm_func(encoded_states)

    # Memory-efficient operations for large batches
    def encode_chunked(self, s: jnp.ndarray, chunk_size: int = 2**18) -> jnp.ndarray:
        """Encode large arrays in chunks for memory efficiency.

        Args:
            s: States to encode
            chunk_size: Size of each chunk

        Returns:
            Encoded states
        """
        if s.shape[0] <= chunk_size:
            return self.encode(s)

        # Split into chunks and process
        num_chunks = (s.shape[0] + chunk_size - 1) // chunk_size
        chunks = jnp.array_split(s, num_chunks, axis=0)

        # Process each chunk and concatenate results
        results = [self.encode(chunk) for chunk in chunks]

        return jnp.concatenate(results, axis=0)

    def decode_chunked(self, encoded: jnp.ndarray, chunk_size: int = 2**18) -> jnp.ndarray:
        """Decode large arrays in chunks for memory efficiency.

        Args:
            encoded: Encoded states to decode
            chunk_size: Size of each chunk

        Returns:
            Decoded states
        """
        if encoded.shape[0] <= chunk_size:
            return self.decode(encoded)

        # Split into chunks and process
        num_chunks = (encoded.shape[0] + chunk_size - 1) // chunk_size
        chunks = jnp.array_split(encoded, num_chunks, axis=0)

        # Process each chunk and concatenate results
        results = [self.decode(chunk) for chunk in chunks]

        return jnp.concatenate(results, axis=0)

    # Utility methods for compatibility
    def get_encoded_length(self) -> int:
        """Get the encoded length for this encoder configuration."""
        return self.encoded_length

    def get_code_width(self) -> int:
        """Get the code width for this encoder configuration."""
        return self.w

    def get_n(self) -> int:
        """Get the string length for this encoder configuration."""
        return self.n

    def validate_input(self, s: jnp.ndarray) -> None:
        """Validate input array for encoding.

        Args:
            s: Input array to validate

        Raises:
            AssertionError: If input is invalid
        """
        assert len(s.shape) == 2, f"Expected 2D array, got shape {s.shape}"
        assert s.shape[1] == self.n, f"Expected second dimension {self.n}, got {s.shape[1]}"
        assert jnp.min(s) >= 0, "Cannot encode negative values"
        max_value = jnp.max(s)
        assert max_value < 2**self.w, f"Width {self.w} is not sufficient to encode value {max_value}"

    def validate_encoded_input(self, encoded: jnp.ndarray) -> None:
        """Validate encoded input array for decoding.

        Args:
            encoded: Encoded array to validate

        Raises:
            AssertionError: If input is invalid
        """
        assert len(encoded.shape) == 2, f"Expected 2D array, got shape {encoded.shape}"
        assert (
            encoded.shape[1] == self.encoded_length
        ), f"Expected second dimension {self.encoded_length}, got {encoded.shape[1]}"
