"""Tests for JAX StringEncoder implementation."""

import math
import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from cayleypy.permutation_utils import apply_permutation

if JAX_AVAILABLE:
    from cayleypy.jax_string_encoder import JAXStringEncoder

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_jax_encode_decode(code_width, n):
    """Test JAX encode/decode functionality."""
    num_states = 5
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)))
    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)

    # Check encoded shape
    expected_encoded_length = int(math.ceil(code_width * n / 63))
    assert s_encoded.shape == (num_states, expected_encoded_length)

    # Check decode recovers original
    s_decoded = enc.decode(s_encoded)
    assert jnp.array_equal(s, s_decoded)


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_jax_permutation(code_width: int, n: int):
    """Test JAX permutation functionality."""
    num_states = 5
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)), dtype=jnp.int64)
    perm = [int(x) for x in np.random.permutation(n)]
    expected = jnp.array([apply_permutation(perm, row) for row in s], dtype=jnp.int64)

    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    perm_func = enc.implement_permutation(perm)
    result_encoded = perm_func(s_encoded)
    result = enc.decode(result_encoded)

    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30)])
def test_jax_permutation_1d(code_width: int, n: int):
    """Test JAX 1D permutation functionality."""
    num_states = 5
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)), dtype=jnp.int64)
    perm = [int(x) for x in np.random.permutation(n)]
    expected = jnp.array([apply_permutation(perm, row) for row in s], dtype=jnp.int64)

    enc = JAXStringEncoder(code_width=code_width, n=n)

    # Only test 1D permutation if encoded_length == 1
    if enc.encoded_length == 1:
        perm_func = enc.implement_permutation_1d(perm)
        s_encoded = enc.encode(s)
        result_encoded = perm_func(s_encoded.squeeze())
        result = enc.decode(result_encoded.reshape(-1, 1))
        assert jnp.array_equal(result, expected)


def test_jax_batch_operations():
    """Test vectorized batch operations."""
    code_width, n = 2, 10
    num_states = 100
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)))

    enc = JAXStringEncoder(code_width=code_width, n=n)

    # Test batch encoding/decoding
    s_encoded = enc.encode(s)
    s_decoded = enc.decode(s_encoded)
    assert jnp.array_equal(s, s_decoded)

    # Test batch permutation
    perm = [int(x) for x in np.random.permutation(n)]
    expected = jnp.array([apply_permutation(perm, row) for row in s])
    result = enc.decode(enc.apply_permutation_to_batch(s_encoded, perm))
    assert jnp.array_equal(result, expected)


def test_jax_chunked_operations():
    """Test memory-efficient chunked operations."""
    code_width, n = 1, 20
    num_states = 1000  # Large enough to test chunking
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)))

    enc = JAXStringEncoder(code_width=code_width, n=n)

    # Test chunked encoding/decoding
    chunk_size = 100
    s_encoded_chunked = enc.encode_chunked(s, chunk_size=chunk_size)
    s_decoded_chunked = enc.decode_chunked(s_encoded_chunked, chunk_size=chunk_size)

    # Compare with regular operations
    s_encoded_regular = enc.encode(s)
    s_decoded_regular = enc.decode(s_encoded_regular)

    assert jnp.array_equal(s_encoded_chunked, s_encoded_regular)
    assert jnp.array_equal(s_decoded_chunked, s_decoded_regular)
    assert jnp.array_equal(s, s_decoded_chunked)


def test_jax_encoder_properties():
    """Test encoder property methods."""
    code_width, n = 3, 15
    enc = JAXStringEncoder(code_width=code_width, n=n)

    assert enc.get_code_width() == code_width
    assert enc.get_n() == n
    assert enc.get_encoded_length() == int(math.ceil(code_width * n / 63))


def test_jax_input_validation():
    """Test input validation methods."""
    code_width, n = 2, 10
    enc = JAXStringEncoder(code_width=code_width, n=n)

    # Valid input
    valid_s = jnp.array(np.random.randint(0, 2**code_width, (5, n)))
    enc.validate_input(valid_s)  # Should not raise

    # Invalid shape
    with pytest.raises(AssertionError):
        invalid_s = jnp.array(np.random.randint(0, 2**code_width, (5, n + 1)))
        enc.validate_input(invalid_s)

    # Negative values
    with pytest.raises(AssertionError):
        invalid_s = jnp.array([[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
        enc.validate_input(invalid_s)

    # Values too large
    with pytest.raises(AssertionError):
        invalid_s = jnp.array([[2**code_width, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
        enc.validate_input(invalid_s)


def test_jax_encoded_input_validation():
    """Test encoded input validation."""
    code_width, n = 2, 10
    enc = JAXStringEncoder(code_width=code_width, n=n)

    # Valid encoded input
    valid_encoded = jnp.zeros((5, enc.encoded_length), dtype=jnp.int64)
    enc.validate_encoded_input(valid_encoded)  # Should not raise

    # Invalid shape
    with pytest.raises(AssertionError):
        invalid_encoded = jnp.zeros((5, enc.encoded_length + 1), dtype=jnp.int64)
        enc.validate_encoded_input(invalid_encoded)


@pytest.mark.parametrize("code_width,n", [(1, 63), (2, 31), (3, 21)])
def test_jax_edge_cases(code_width, n):
    """Test edge cases for different code widths and lengths."""
    num_states = 10
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)))

    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    s_decoded = enc.decode(s_encoded)

    assert jnp.array_equal(s, s_decoded)

    # Test permutation
    perm = [int(x) for x in np.random.permutation(n)]
    expected = jnp.array([apply_permutation(perm, row) for row in s])
    result = enc.decode(enc.apply_permutation_to_batch(s_encoded, perm))
    assert jnp.array_equal(result, expected)


def test_jax_single_element():
    """Test encoding/decoding single elements."""
    code_width, n = 4, 1
    s = jnp.array([[7], [3], [15], [0]])  # Single elements

    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    s_decoded = enc.decode(s_encoded)

    assert jnp.array_equal(s, s_decoded)


def test_jax_identity_permutation():
    """Test identity permutation (should not change anything)."""
    code_width, n = 2, 8
    num_states = 5
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)))

    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)

    # Identity permutation
    identity_perm = list(range(n))
    perm_func = enc.implement_permutation(identity_perm)
    result_encoded = perm_func(s_encoded)
    result = enc.decode(result_encoded)

    assert jnp.array_equal(s, result)


def test_jax_reverse_permutation():
    """Test reverse permutation."""
    code_width, n = 1, 10
    num_states = 5
    s = jnp.array(np.random.randint(0, 2**code_width, (num_states, n)))

    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)

    # Reverse permutation
    reverse_perm = list(range(n - 1, -1, -1))
    expected = jnp.array([apply_permutation(reverse_perm, row) for row in s])
    result = enc.decode(enc.apply_permutation_to_batch(s_encoded, reverse_perm))

    assert jnp.array_equal(result, expected)


# Compatibility tests with PyTorch implementation (if available)
def test_jax_pytorch_compatibility():
    """Test compatibility between JAX and PyTorch implementations."""
    try:
        import torch
        from .string_encoder import StringEncoder

        code_width, n = 2, 15
        num_states = 10

        # Create same random data for both implementations
        np.random.seed(42)
        s_np = np.random.randint(0, 2**code_width, (num_states, n))
        s_torch = torch.tensor(s_np, dtype=torch.int64)
        s_jax = jnp.array(s_np, dtype=jnp.int64)

        # Create encoders
        torch_enc = StringEncoder(code_width=code_width, n=n)
        jax_enc = JAXStringEncoder(code_width=code_width, n=n)

        # Test encoding
        torch_encoded = torch_enc.encode(s_torch)
        jax_encoded = jax_enc.encode(s_jax)

        # Convert to numpy for comparison
        torch_encoded_np = torch_encoded.numpy()
        jax_encoded_np = np.array(jax_encoded)

        assert np.array_equal(torch_encoded_np, jax_encoded_np)

        # Test decoding
        torch_decoded = torch_enc.decode(torch_encoded)
        jax_decoded = jax_enc.decode(jax_encoded)

        torch_decoded_np = torch_decoded.numpy()
        jax_decoded_np = np.array(jax_decoded)

        assert np.array_equal(torch_decoded_np, jax_decoded_np)
        assert np.array_equal(s_np, jax_decoded_np)

        # Test permutation
        perm = [int(x) for x in np.random.permutation(n)]

        # PyTorch permutation
        torch_encoded_result = torch.zeros_like(torch_encoded)
        torch_perm_func = torch_enc.implement_permutation(perm)
        torch_perm_func(torch_encoded, torch_encoded_result)
        torch_result = torch_enc.decode(torch_encoded_result)

        # JAX permutation
        jax_perm_func = jax_enc.implement_permutation(perm)
        jax_encoded_result = jax_perm_func(jax_encoded)
        jax_result = jax_enc.decode(jax_encoded_result)

        # Compare results
        torch_result_np = torch_result.numpy()
        jax_result_np = np.array(jax_result)

        assert np.array_equal(torch_result_np, jax_result_np)

    except ImportError:
        pytest.skip("PyTorch not available for compatibility testing")


if __name__ == "__main__":
    pytest.main([__file__])
