"""Tests for JAX string encoder implementation."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from .jax_string_encoder import JAXStringEncoder
from .permutation_utils import apply_permutation
from .string_encoder import StringEncoder


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_jax_encode_decode(code_width, n):
    """Test JAX encoder encode/decode functionality."""
    num_states = 5
    s_np = np.random.randint(0, 2**code_width, (num_states, n))
    s_jax = jnp.array(s_np, dtype=jnp.int64)
    
    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s_jax)
    
    assert s_encoded.shape == (num_states, int(math.ceil(code_width * n / 63)))
    decoded = enc.decode(s_encoded)
    
    assert jnp.array_equal(s_jax, decoded)


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_jax_pytorch_equivalence(code_width, n):
    """Test that JAX and PyTorch encoders produce equivalent results."""
    num_states = 5
    s_np = np.random.randint(0, 2**code_width, (num_states, n))
    s_torch = torch.tensor(s_np, dtype=torch.int64)
    s_jax = jnp.array(s_np, dtype=jnp.int64)
    
    # Create encoders
    torch_enc = StringEncoder(code_width=code_width, n=n)
    jax_enc = JAXStringEncoder(code_width=code_width, n=n)
    
    # Test encoding equivalence
    torch_encoded = torch_enc.encode(s_torch)
    jax_encoded = jax_enc.encode(s_jax)
    
    assert torch_encoded.shape == jax_encoded.shape
    np.testing.assert_array_equal(torch_encoded.numpy(), np.array(jax_encoded))
    
    # Test decoding equivalence
    torch_decoded = torch_enc.decode(torch_encoded)
    jax_decoded = jax_enc.decode(jax_encoded)
    
    np.testing.assert_array_equal(torch_decoded.numpy(), np.array(jax_decoded))


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_jax_permutation(code_width: int, n: int):
    """Test JAX permutation implementation."""
    num_states = 5
    s_np = np.random.randint(0, 2**code_width, (num_states, n))
    s_jax = jnp.array(s_np, dtype=jnp.int64)
    perm = [int(x) for x in np.random.permutation(n)]
    expected = jnp.array([apply_permutation(perm, row) for row in s_np], dtype=jnp.int64)
    
    enc = JAXStringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s_jax)
    perm_func = enc.implement_permutation(perm)
    result_encoded = perm_func(s_encoded)
    result = enc.decode(result_encoded)
    
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30)])
def test_jax_permutation_1d(code_width: int, n: int):
    """Test JAX 1D permutation implementation."""
    num_states = 5
    s_np = np.random.randint(0, 2**code_width, (num_states, n))
    s_jax = jnp.array(s_np, dtype=jnp.int64)
    perm = [int(x) for x in np.random.permutation(n)]
    expected = jnp.array([apply_permutation(perm, row) for row in s_np], dtype=jnp.int64)
    
    enc = JAXStringEncoder(code_width=code_width, n=n)
    perm_func = enc.implement_permutation_1d(perm)
    s_encoded = enc.encode(s_jax)
    result_encoded = perm_func(s_encoded.squeeze())  # 1D requires squeeze
    result = enc.decode(result_encoded.reshape(num_states, -1))
    
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30)])
def test_jax_pytorch_permutation_equivalence(code_width: int, n: int):
    """Test that JAX and PyTorch permutation implementations produce equivalent results."""
    num_states = 5
    s_np = np.random.randint(0, 2**code_width, (num_states, n))
    s_torch = torch.tensor(s_np, dtype=torch.int64)
    s_jax = jnp.array(s_np, dtype=jnp.int64)
    perm = [int(x) for x in np.random.permutation(n)]
    
    # PyTorch implementation
    torch_enc = StringEncoder(code_width=code_width, n=n)
    torch_encoded = torch_enc.encode(s_torch)
    torch_result = torch.zeros_like(torch_encoded)
    torch_perm_func = torch_enc.implement_permutation(perm)
    torch_perm_func(torch_encoded, torch_result)
    torch_decoded = torch_enc.decode(torch_result)
    
    # JAX implementation
    jax_enc = JAXStringEncoder(code_width=code_width, n=n)
    jax_encoded = jax_enc.encode(s_jax)
    jax_perm_func = jax_enc.implement_permutation(perm)
    jax_result = jax_perm_func(jax_encoded)
    jax_decoded = jax_enc.decode(jax_result)
    
    # Compare results
    np.testing.assert_array_equal(torch_decoded.numpy(), np.array(jax_decoded))


def test_jax_batch_operations():
    """Test vectorized batch operations using vmap."""
    code_width, n = 2, 10
    num_batches = 3
    batch_size = 5
    
    # Create batch of data
    batches = []
    for _ in range(num_batches):
        batch = np.random.randint(0, 2**code_width, (batch_size, n))
        batches.append(batch)
    
    batch_data = jnp.array(batches, dtype=jnp.int64)
    
    enc = JAXStringEncoder(code_width=code_width, n=n)
    
    # Test batch encoding
    encoded_batches = enc.encode_batch(batch_data)
    assert encoded_batches.shape[0] == num_batches
    assert encoded_batches.shape[1] == batch_size
    
    # Test batch decoding
    decoded_batches = enc.decode_batch(encoded_batches)
    assert jnp.array_equal(batch_data, decoded_batches)


def test_jax_error_handling():
    """Test error handling in JAX string encoder."""
    enc = JAXStringEncoder(code_width=2, n=5)
    
    # Test negative values
    s_negative = jnp.array([[-1, 0, 1, 2, 3]], dtype=jnp.int64)
    with pytest.raises(AssertionError, match="Cannot encode negative values"):
        enc.encode(s_negative)
    
    # Test values too large for code width
    s_large = jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int64)  # 4 requires 3 bits, but code_width=2
    with pytest.raises(AssertionError, match="Width 2 is not sufficient"):
        enc.encode(s_large)


def test_jax_encoded_length_calculation():
    """Test that encoded length is calculated correctly."""
    # Test cases where encoded length should be 1
    enc1 = JAXStringEncoder(code_width=1, n=63)
    assert enc1.encoded_length == 1
    
    enc2 = JAXStringEncoder(code_width=1, n=64)
    assert enc2.encoded_length == 2
    
    enc3 = JAXStringEncoder(code_width=2, n=31)
    assert enc3.encoded_length == 1
    
    enc4 = JAXStringEncoder(code_width=2, n=32)
    assert enc4.encoded_length == 2


def test_jax_permutation_identity():
    """Test that identity permutation works correctly."""
    code_width, n = 2, 10
    num_states = 5
    s_np = np.random.randint(0, 2**code_width, (num_states, n))
    s_jax = jnp.array(s_np, dtype=jnp.int64)
    
    enc = JAXStringEncoder(code_width=code_width, n=n)
    identity_perm = list(range(n))
    perm_func = enc.implement_permutation(identity_perm)
    
    s_encoded = enc.encode(s_jax)
    result_encoded = perm_func(s_encoded)
    result = enc.decode(result_encoded)
    
    assert jnp.array_equal(s_jax, result)