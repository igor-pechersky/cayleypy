"""JAX-based string encoder for efficient state representation and manipulation."""

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

# We are using int64, but avoid using the sign bit.
CODEWORD_LENGTH = 63


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
        assert 1 <= code_width <= CODEWORD_LENGTH
        self.w = code_width
        self.n = n
        self.encoded_length = int(math.ceil(self.n * self.w / CODEWORD_LENGTH))  # Encoded length.

    @partial(jax.jit, static_argnums=(0,))
    def encode(self, s: jax.Array) -> jax.Array:
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
        
        # Vectorized encoding using JAX operations
        for i in range(w * self.n):
            bit_pos_in_original = i % w
            element_idx = i // w
            codeword_idx = i // cl
            bit_pos_in_codeword = i % cl
            
            # Extract bit from original array
            bit = (s[:, element_idx] >> bit_pos_in_original) & 1
            # Set bit in encoded array
            encoded = encoded.at[:, codeword_idx].set(
                encoded[:, codeword_idx] | (bit << bit_pos_in_codeword)
            )
        
        return encoded

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, encoded: jax.Array) -> jax.Array:
        """Decodes array of coset elements.

        Input shape `(m, self.encoded_length)`. Output shape `(m, self.n)`.
        """
        orig = jnp.zeros((encoded.shape[0], self.n), dtype=jnp.int64)
        w, cl = self.w, CODEWORD_LENGTH
        
        # Vectorized decoding using JAX operations
        for i in range(w * self.n):
            bit_pos_in_original = i % w
            element_idx = i // w
            codeword_idx = i // cl
            bit_pos_in_codeword = i % cl
            
            # Extract bit from encoded array
            bit = (encoded[:, codeword_idx] >> bit_pos_in_codeword) & 1
            # Set bit in original array
            orig = orig.at[:, element_idx].set(
                orig[:, element_idx] | (bit << bit_pos_in_original)
            )
        
        return orig

    def prepare_shift_to_mask(self, p: list[int]) -> dict[tuple[int, int, int], int]:
        """Prepare shift-to-mask mapping for permutation implementation."""
        assert len(p) == self.n
        shift_to_mask: dict[tuple[int, int, int], int] = {}
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

    def implement_permutation(self, p: list[int]) -> Callable[[jax.Array], jax.Array]:
        """Converts permutation to a JIT-compiled function on encoded array implementing this permutation.

        Returns a function that takes an encoded array and returns the permuted encoded array.
        """
        shift_to_mask = self.prepare_shift_to_mask(p)
        
        @jax.jit
        def permutation_func(x: jax.Array) -> jax.Array:
            y = jnp.zeros_like(x)
            for (start_cw_id, end_cw_id, shift), mask in shift_to_mask.items():
                masked_bits = x[:, start_cw_id] & mask
                if shift > 0:
                    shifted_bits = masked_bits << shift
                elif shift < 0:
                    shifted_bits = masked_bits >> (-shift)
                else:
                    shifted_bits = masked_bits
                y = y.at[:, end_cw_id].set(y[:, end_cw_id] | shifted_bits)
            return y
        
        return permutation_func

    def implement_permutation_1d(self, p: list[int]) -> Callable[[jax.Array], jax.Array]:
        """Converts permutation to a JIT-compiled function on 1D encoded array implementing this permutation.

        The function converts 1D array to 1D array of the same dimension.
        Applicable only if state can be encoded by single int64 (encoded_length=1).
        """
        assert self.encoded_length == 1
        shift_to_mask = self.prepare_shift_to_mask(p)
        
        @jax.jit
        def permutation_func_1d(x: jax.Array) -> jax.Array:
            result = jnp.zeros_like(x)
            for (_, _, shift), mask in shift_to_mask.items():
                masked_bits = x & mask
                if shift > 0:
                    shifted_bits = masked_bits << shift
                elif shift < 0:
                    shifted_bits = masked_bits >> (-shift)
                else:
                    shifted_bits = masked_bits
                result = result | shifted_bits
            return result
        
        return permutation_func_1d

    # Vectorized versions using vmap for batch operations
    @partial(jax.jit, static_argnums=(0,))
    def encode_batch(self, s: jax.Array) -> jax.Array:
        """Vectorized batch encoding using vmap."""
        return jax.vmap(self.encode)(s)

    @partial(jax.jit, static_argnums=(0,))
    def decode_batch(self, encoded: jax.Array) -> jax.Array:
        """Vectorized batch decoding using vmap."""
        return jax.vmap(self.decode)(encoded)