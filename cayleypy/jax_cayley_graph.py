"""JAX implementation of CayleyGraph for TPU/GPU optimization.

This module provides a JAX-based implementation of the CayleyGraph class,
maintaining full API compatibility with the PyTorch version while leveraging
JAX's functional programming paradigm and advanced compilation features.
"""
# pylint: disable=too-many-lines

import gc
import math
from functools import partial
from typing import Optional, Union, List, TYPE_CHECKING, Any, Type

import numpy as np

from .cayley_graph_def import CayleyGraphDef, GeneratorType
from .jax_device_manager import JAXDeviceManager
from .jax_hasher import JAXStateHasher
from .jax_string_encoder import JAXStringEncoder
from .jax_tensor_ops import (
    gather_along_axis,
    isin_via_searchsorted,
    tensor_split,
    concatenate_arrays,
    stack_arrays,
    sort_with_indices,
    ensure_jax_array,
)

if TYPE_CHECKING:
    from .bfs_result import BfsResult as BFS_RESULT_TYPE
    from .jax_hash_set import JAXHashSet as JAX_HASH_SET_TYPE
    from .beam_search_result import BeamSearchResult as BEAM_SEARCH_RESULT_TYPE
    from .predictor import Predictor as PREDICTOR_TYPE
    from jax.sharding import PositionalSharding as POSITIONAL_SHARDING_TYPE
else:
    BFS_RESULT_TYPE = None
    JAX_HASH_SET_TYPE = None
    BEAM_SEARCH_RESULT_TYPE = None
    PREDICTOR_TYPE = None
    POSITIONAL_SHARDING_TYPE = None

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from jax import jit
    from jax.sharding import PositionalSharding

    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    jrandom = None  # type: ignore
    PositionalSharding = None  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Runtime imports for modules that may cause circular imports
BfsResult: Optional[Type[Any]] = None
JAXHashSet: Optional[Type[Any]] = None
BeamSearchResult: Optional[Type[Any]] = None
Predictor: Optional[Type[Any]] = None

try:
    from .bfs_result import BfsResult  # type: ignore
except ImportError:
    pass

try:
    from .jax_hash_set import JAXHashSet  # type: ignore
except ImportError:
    pass

try:
    from .beam_search_result import BeamSearchResult  # type: ignore
    from .predictor import Predictor  # type: ignore
except ImportError:
    pass

# PositionalSharding is now imported with the main JAX imports above


def _check_jax_available():
    """Check if JAX is available and raise error if not."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not available. Install with: pip install 'cayleypy[jax-cpu]', "
            "'cayleypy[jax-cuda]', or 'cayleypy[jax-tpu]'"
        )


class JAXCayleyGraph:
    """JAX-based Cayley graph implementation with identical API to PyTorch version.

    This class provides a JAX backend for CayleyGraph operations, optimized for
    TPU/GPU computation while maintaining full compatibility with the PyTorch API.

    Key features:
    - Automatic device management (TPU > GPU > CPU)
    - JIT compilation for performance-critical operations
    - Vectorized operations using JAX
    - Memory-efficient chunked processing
    - Functional programming paradigm
    """

    def __init__(
        self,
        definition: CayleyGraphDef,
        *,
        device: str = "auto",
        random_seed: Optional[int] = None,
        bit_encoding_width: Union[Optional[int], str] = "auto",
        verbose: int = 0,
        batch_size: int = 2**20,
        hash_chunk_size: int = 2**25,
        memory_limit_gb: float = 16,
    ):
        """Initialize JAX CayleyGraph.

        Args:
            definition: Definition of the graph (as CayleyGraphDef)
            device: Device preference ("auto", "tpu", "gpu", "cpu")
            random_seed: Random seed for deterministic hashing
            bit_encoding_width: Bits to encode one element ("auto" or int or None)
            verbose: Level of logging (0 means no logging)
            batch_size: Size of batch for batch processing
            hash_chunk_size: Size of chunk for hashing
            memory_limit_gb: Approximate available memory in GB
        """
        _check_jax_available()

        self.definition = definition
        self.verbose = verbose
        self.batch_size = batch_size
        self.memory_limit_bytes = int(memory_limit_gb * (2**30))

        # Initialize device manager
        self.device_manager = JAXDeviceManager(device)
        if verbose > 0:
            print(f"Using device: {self.device_manager.primary_device}")

        # TPU-specific configuration
        self.is_tpu = self.device_manager.is_tpu()
        self.num_devices = len(self.device_manager.devices) if self.is_tpu else 1
        self.tpu_shard_threshold = 2**20  # Shard arrays larger than 1M elements

        # Set up central state
        self.central_state = self.device_manager.put_on_device(jnp.array(definition.central_state, dtype=jnp.int64))

        # Initialize encoding system
        self.encoded_state_size = self.definition.state_size
        self.string_encoder: Optional[JAXStringEncoder] = None

        if definition.is_permutation_group():
            # Convert permutations to JAX arrays
            self.permutations_jax = self.device_manager.put_on_device(
                jnp.array(definition.generators_permutations, dtype=jnp.int64)
            )

            # Set up bit encoding if requested
            if bit_encoding_width == "auto":
                max_element = int(jnp.max(self.central_state))
                required_bits = int(math.ceil(math.log2(max_element + 1)))
                # Only use string encoding if it provides significant benefit
                # For small groups, the overhead isn't worth it
                if required_bits < 6 and self.definition.state_size <= 10:
                    bit_encoding_width = None
                else:
                    bit_encoding_width = required_bits

            if bit_encoding_width is not None:
                self.string_encoder = JAXStringEncoder(code_width=int(bit_encoding_width), n=self.definition.state_size)
                # Pre-compile encoded generators for performance
                self.encoded_generators = [
                    self.string_encoder.implement_permutation(perm) for perm in definition.generators_permutations
                ]
                self.encoded_state_size = self.string_encoder.encoded_length

        # Initialize hasher
        self.hasher = JAXStateHasher(
            state_size=self.encoded_state_size,
            random_seed=random_seed,
            chunk_size=hash_chunk_size,
            use_string_encoder=(self.string_encoder is not None),
        )

        # Compute central state hash
        encoded_central = self.encode_states(self.central_state)
        self.central_state_hash = self.hasher.hash_states(encoded_central)

    def _get_unique_states(
        self, states: jnp.ndarray, hashes: Optional[jnp.ndarray] = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Remove duplicates from states and sort them by hash.

        Args:
            states: Input states
            hashes: Optional pre-computed hashes

        Returns:
            Tuple of (unique_states, unique_hashes)
        """
        if self.hasher.is_identity:
            unique_hashes = jnp.unique(states.reshape(-1))
            return unique_hashes.reshape((-1, 1)), unique_hashes

        if hashes is None:
            hashes = self.hasher.hash_states(states)

        # Sort by hash values
        hashes_sorted, idx = sort_with_indices(hashes, stable=True)

        # Find unique values using mask
        mask = jnp.ones(hashes_sorted.shape[0], dtype=bool)
        if hashes_sorted.shape[0] > 1:
            mask = mask.at[1:].set(hashes_sorted[1:] != hashes_sorted[:-1])

        unique_idx = idx[mask]
        return states[unique_idx], hashes[unique_idx]

    def encode_states(self, states: Union[jnp.ndarray, np.ndarray, list]) -> jnp.ndarray:
        """Convert states from human-readable to internal representation.

        Args:
            states: States to encode

        Returns:
            Encoded states as JAX array
        """
        states = ensure_jax_array(states)
        states = self.device_manager.put_on_device(states)

        # Handle matrix groups - states can be passed as 2D matrices or flattened vectors
        if self.definition.is_matrix_group():
            n, m = self.definition.decoded_state_shape
            if states.ndim == 2 and states.shape == (n, m):
                # Single matrix - flatten it
                states = states.reshape(1, -1)
            elif states.ndim == 3:
                # Batch of matrices - flatten each
                states = states.reshape(states.shape[0], -1)
            elif states.ndim == 2 and states.shape[1] == self.definition.state_size:
                # Already flattened batch format
                pass
            elif states.ndim == 1 and len(states) == self.definition.state_size:
                # Single flattened state
                states = states.reshape(1, -1)
            else:
                # Try to reshape as matrices and then flatten
                try:
                    if states.size % (n * m) == 0:
                        num_matrices = states.size // (n * m)
                        states = states.reshape(num_matrices, n, m).reshape(num_matrices, -1)
                    else:
                        raise ValueError(f"Cannot reshape state to {n}x{m} matrix")
                except Exception as exc:
                    raise ValueError(
                        f"Invalid matrix state shape: expected {n}x{m} matrix or "
                        f"flattened {n*m} vector, got {states.shape}"
                    ) from exc
        else:
            # Handle permutation groups
            if states.ndim == 1:
                states = states.reshape(1, -1)
            elif states.ndim > 2:
                states = states.reshape(-1, self.definition.state_size)

        # Validate state dimensions
        if states.shape[1] != self.definition.state_size:
            raise ValueError(f"State size mismatch: expected {self.definition.state_size}, got {states.shape[1]}")

        if self.string_encoder is not None:
            return self.string_encoder.encode(states)
        return states

    def decode_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Convert states from internal to human-readable representation.

        Args:
            states: Encoded states

        Returns:
            Decoded states
        """
        if self.definition.generators_type == GeneratorType.MATRIX:
            n, m = self.definition.decoded_state_shape
            # Internally states are vectors, but mathematically they are n*m matrices
            return states.reshape((-1, n, m))

        if self.string_encoder is not None:
            return self.string_encoder.decode(states)
        return states

    @partial(jit, static_argnums=(0, 1))
    def _apply_generator_batched_permutation_encoded(self, gen_idx: int, src: jnp.ndarray) -> jnp.ndarray:
        """Apply generator to encoded permutation states (JIT compiled).

        Args:
            gen_idx: Generator index
            src: Source states

        Returns:
            States after applying generator
        """
        return self.encoded_generators[gen_idx](src)

    @partial(jit, static_argnums=(0, 1))
    def _apply_generator_batched_permutation_regular(self, gen_idx: int, src: jnp.ndarray) -> jnp.ndarray:
        """Apply generator to regular permutation states (JIT compiled).

        Args:
            gen_idx: Generator index
            src: Source states

        Returns:
            States after applying generator
        """
        states_num = src.shape[0]
        move = self.permutations_jax[gen_idx].reshape((1, -1))
        move = jnp.tile(move, (states_num, 1))
        return gather_along_axis(src, move, axis=1)

    def _apply_generator_batched_matrix(self, gen_idx: int, src: jnp.ndarray) -> jnp.ndarray:
        """Apply matrix generator to states.

        Args:
            gen_idx: Generator index
            src: Source states

        Returns:
            States after applying generator
        """
        states_num = src.shape[0]
        n, m = self.definition.decoded_state_shape
        mx = self.definition.generators_matrices[gen_idx]

        # Reshape states to matrices
        src_matrices = src.reshape((states_num, n, m))

        # Apply matrix multiplication
        mx_jax = self.device_manager.put_on_device(jnp.array(mx.matrix, dtype=jnp.int64))

        # Vectorized matrix multiplication
        result = jnp.matmul(mx_jax[None, :, :], src_matrices)

        # Apply modulo if needed
        if mx.modulo > 0:
            result = result % mx.modulo

        # Reshape back to vector format
        return result.reshape((states_num, n * m))

    def _apply_generator_batched(self, gen_idx: int, src: jnp.ndarray) -> jnp.ndarray:
        """Apply generator to batch of states.

        Args:
            gen_idx: Generator index
            src: Source states

        Returns:
            States after applying generator
        """
        if self.definition.is_permutation_group():
            if self.string_encoder is not None:
                return self._apply_generator_batched_permutation_encoded(gen_idx, src)
            else:
                return self._apply_generator_batched_permutation_regular(gen_idx, src)
        else:
            return self._apply_generator_batched_matrix(gen_idx, src)

    def _validate_state(self, state: Union[jnp.ndarray, np.ndarray, list]) -> bool:
        """Validate that a state is valid for this graph.

        Args:
            state: State to validate

        Returns:
            True if state is valid, False otherwise
        """
        try:
            state_array = jnp.array(state)

            # Check length
            if len(state_array) != self.definition.state_size:
                return False

            # For permutation groups, check that it's a valid permutation
            if self.definition.is_permutation_group():
                # Check that all elements are in valid range
                if jnp.any(state_array < 0) or jnp.any(state_array >= self.definition.state_size):
                    return False

                # Check that all elements are unique (valid permutation)
                if len(jnp.unique(state_array)) != len(state_array):
                    return False

            return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False

    def apply_generator(self, states: Union[jnp.ndarray, np.ndarray, list], generator_id: int) -> jnp.ndarray:
        """Apply a single generator to given state(s).

        Args:
            states: One or more states to apply generator to
            generator_id: Index of generator to apply

        Returns:
            States after applying specified generator
        """
        if generator_id < 0 or generator_id >= self.definition.n_generators:
            raise IndexError(f"Generator index {generator_id} out of range [0, {self.definition.n_generators})")

        # Check if input is a single state
        input_states = ensure_jax_array(states)
        is_single_state = False

        if self.definition.is_matrix_group():
            n, m = self.definition.decoded_state_shape
            # Single matrix input (2D) or single flattened state (1D)
            if (input_states.ndim == 2 and input_states.shape == (n, m)) or (
                input_states.ndim == 1 and len(input_states) == self.definition.state_size
            ):
                is_single_state = True
        else:
            # Permutation group - single state is 1D array
            if input_states.ndim == 1 and len(input_states) == self.definition.state_size:
                is_single_state = True

        states = self.encode_states(states)
        result = self._apply_generator_batched(generator_id, states)
        decoded_result = self.decode_states(result)

        # If input was a single state, return a single state (remove batch dimension)
        if is_single_state and decoded_result.shape[0] == 1:
            return decoded_result[0]
        return decoded_result

    @partial(jit, static_argnums=(0, 2))
    def _apply_path_compiled_permutation(self, states: jnp.ndarray, generator_ids: tuple) -> jnp.ndarray:
        """Apply path with JIT compilation for permutation groups.

        Args:
            states: Input states (encoded)
            generator_ids: Tuple of generator indices (must be tuple for JIT)

        Returns:
            States after applying generators in order
        """
        current_states = states

        for gen_id in generator_ids:
            if self.string_encoder is not None:
                # Use encoded generators
                current_states = self.encoded_generators[gen_id](current_states)
            else:
                # Use regular permutation application
                states_num = current_states.shape[0]
                move = self.permutations_jax[gen_id].reshape((1, -1))
                move = jnp.tile(move, (states_num, 1))
                current_states = gather_along_axis(current_states, move, axis=1)

        return current_states

    @partial(jit, static_argnums=(0, 2))
    def _apply_path_compiled_matrix(self, states: jnp.ndarray, generator_ids: tuple) -> jnp.ndarray:
        """Apply path with JIT compilation for matrix groups.

        Args:
            states: Input states (encoded, flattened matrices)
            generator_ids: Tuple of generator indices (must be tuple for JIT)

        Returns:
            States after applying generators in order
        """
        current_states = states
        states_num = current_states.shape[0]
        n, m = self.definition.decoded_state_shape

        for gen_id in generator_ids:
            # Reshape to matrices
            states_matrices = current_states.reshape((states_num, n, m))

            # Get generator matrix
            mx = self.definition.generators_matrices[gen_id]
            mx_jax = self.device_manager.put_on_device(jnp.array(mx.matrix, dtype=jnp.int64))

            # Apply matrix multiplication
            result = jnp.matmul(mx_jax[None, :, :], states_matrices)

            # Apply modulo if needed
            if mx.modulo > 0:
                result = result % mx.modulo

            # Flatten back
            current_states = result.reshape((states_num, n * m))

        return current_states

    def apply_path(self, states: Union[jnp.ndarray, np.ndarray, list], generator_ids: List[int]) -> jnp.ndarray:
        """Apply multiple generators to given state(s) in order.

        Args:
            states: One or more states to apply generators to
            generator_ids: Indices of generators to apply in order

        Returns:
            States after applying specified generators in order
        """
        # Check if input is a single state for proper output formatting
        input_states = ensure_jax_array(states)
        is_single_state = False

        if self.definition.is_matrix_group():
            n, m = self.definition.decoded_state_shape
            if (input_states.ndim == 2 and input_states.shape == (n, m)) or (
                input_states.ndim == 1 and len(input_states) == self.definition.state_size
            ):
                is_single_state = True
        else:
            if input_states.ndim == 1 and len(input_states) == self.definition.state_size:
                is_single_state = True

        # Validate generator IDs
        for gen_id in generator_ids:
            if gen_id < 0 or gen_id >= self.definition.n_generators:
                raise IndexError(f"Generator index {gen_id} out of range [0, {self.definition.n_generators})")

        encoded_states = self.encode_states(states)

        # Use JIT-compiled versions for better performance
        # Convert list to tuple for JIT compatibility
        generator_ids_tuple = tuple(generator_ids)

        if self.definition.is_permutation_group():
            result_states = self._apply_path_compiled_permutation(encoded_states, generator_ids_tuple)
        elif self.definition.is_matrix_group():
            result_states = self._apply_path_compiled_matrix(encoded_states, generator_ids_tuple)
        else:
            # Fallback to sequential application
            result_states = encoded_states
            for gen_id in generator_ids:
                result_states = self._apply_generator_batched(gen_id, result_states)

        decoded_result = self.decode_states(result_states)

        # If input was a single state, return a single state (remove batch dimension)
        if is_single_state and decoded_result.shape[0] == 1:
            return decoded_result[0]

        return decoded_result

    @partial(jit, static_argnums=(0,))
    def _get_neighbors_compiled_permutation(self, states: jnp.ndarray) -> jnp.ndarray:
        """Get neighbors with JIT compilation for permutation groups.

        Args:
            states: Input states

        Returns:
            All neighbors of input states
        """
        states_num = states.shape[0]
        n_generators = self.definition.n_generators

        # Pre-allocate result array
        neighbors = jnp.zeros((states_num * n_generators, states.shape[1]), dtype=jnp.int64)

        # Apply each generator
        for i in range(n_generators):
            start_idx = i * states_num
            end_idx = (i + 1) * states_num

            if self.string_encoder is not None:
                # Use encoded generators
                result = self.encoded_generators[i](states)
            else:
                # Use regular permutation application
                move = self.permutations_jax[i].reshape((1, -1))
                move = jnp.tile(move, (states_num, 1))
                result = gather_along_axis(states, move, axis=1)

            neighbors = neighbors.at[start_idx:end_idx].set(result)

        return neighbors

    @partial(jit, static_argnums=(0,))
    def _get_neighbors_compiled_matrix(self, states: jnp.ndarray) -> jnp.ndarray:
        """Get neighbors with JIT compilation for matrix groups.

        Args:
            states: Input states (flattened matrices)

        Returns:
            All neighbors of input states
        """
        states_num = states.shape[0]
        n_generators = self.definition.n_generators
        n, m = self.definition.decoded_state_shape

        # Pre-allocate result array
        neighbors = jnp.zeros((states_num * n_generators, states.shape[1]), dtype=jnp.int64)

        # Reshape states to matrices for computation
        states_matrices = states.reshape((states_num, n, m))

        # Apply each generator
        for i in range(n_generators):
            start_idx = i * states_num
            end_idx = (i + 1) * states_num

            mx = self.definition.generators_matrices[i]
            mx_jax = self.device_manager.put_on_device(jnp.array(mx.matrix, dtype=jnp.int64))

            # Vectorized matrix multiplication
            result = jnp.matmul(mx_jax[None, :, :], states_matrices)

            # Apply modulo if needed
            if mx.modulo > 0:
                result = result % mx.modulo

            # Reshape back to vector format and store
            result_flat = result.reshape((states_num, n * m))
            neighbors = neighbors.at[start_idx:end_idx].set(result_flat)

        return neighbors

    def get_neighbors(self, states: jnp.ndarray) -> jnp.ndarray:
        """Calculate all neighbors of states (in internal representation).

        Args:
            states: Input states in internal representation

        Returns:
            All neighbors of input states
        """
        # Use JIT-compiled versions for better performance
        if self.definition.is_permutation_group():
            return self._get_neighbors_compiled_permutation(states)
        elif self.definition.is_matrix_group():
            return self._get_neighbors_compiled_matrix(states)
        else:
            # Fallback to general implementation
            states_num = states.shape[0]
            neighbors = jnp.zeros((states_num * self.definition.n_generators, states.shape[1]), dtype=jnp.int64)

            for i in range(self.definition.n_generators):
                start_idx = i * states_num
                end_idx = (i + 1) * states_num
                result = self._apply_generator_batched(i, states)
                neighbors = neighbors.at[start_idx:end_idx].set(result)

            return neighbors

    def get_neighbors_decoded(self, states: jnp.ndarray) -> jnp.ndarray:
        """Calculate neighbors in decoded (external) representation.

        Args:
            states: Input states in external representation

        Returns:
            Neighbors in external representation
        """
        encoded_states = self.encode_states(states)
        neighbors = self.get_neighbors(encoded_states)
        return self.decode_states(neighbors)

    def bfs(
        self,
        *,
        start_states: Union[None, jnp.ndarray, np.ndarray, list] = None,
        max_layer_size_to_store: Optional[int] = 1000,
        max_layer_size_to_explore: int = 10**9,
        max_diameter: int = 1000000,
        return_all_edges: bool = False,
        return_all_hashes: bool = False,
        enable_tpu_sharding: bool = True,
    ):
        """Run breadth-first search (BFS) algorithm from given start_states.

        BFS visits all vertices of the graph in layers, where next layer contains vertices adjacent to previous layer
        that were not visited before. As a result, we get all vertices grouped by their distance from the set of initial
        states.

        Args:
            start_states: States on 0-th layer of BFS. Defaults to central state of the graph.
            max_layer_size_to_store: Maximal size of layer to store.
                If None, all layers will be stored (use this if you need full graph).
                Defaults to 1000. First and last layers are always stored.
            max_layer_size_to_explore: If reaches layer of larger size, will stop the BFS.
            max_diameter: Maximal number of BFS iterations.
            return_all_edges: Whether to return list of all edges (uses more memory).
            return_all_hashes: Whether to return hashes for all vertices (uses more memory).

        Returns:
            BfsResult object with requested BFS results.
        """
        # Use imported BfsResult
        if BfsResult is None:
            raise ImportError("BfsResult not available")

        start_states = self.encode_states(start_states or self.central_state)
        layer1, layer1_hashes = self._get_unique_states(start_states)
        layer_sizes = [len(layer1)]
        layers = {0: self.decode_states(layer1)}
        full_graph_explored = False
        edges_list_starts: list[jnp.ndarray] = []
        edges_list_ends: list[jnp.ndarray] = []
        all_layers_hashes: list[jnp.ndarray] = []
        max_layer_size_to_store = max_layer_size_to_store or 10**15

        # When we don't need edges, we can apply more memory-efficient algorithm with batching.
        # This algorithm finds neighbors in batches and removes duplicates from batches before stacking them.
        do_batching = not return_all_edges

        # Stores hashes of previous layers, so BFS does not visit already visited states again.
        # If generators are inverse closed, only 2 last layers are stored here.
        seen_states_hashes = [layer1_hashes]

        # Returns mask where 0s are at positions in `current_layer_hashes` that were seen previously.
        def _remove_seen_states(current_layer_hashes: jnp.ndarray) -> jnp.ndarray:
            ans = ~isin_via_searchsorted(current_layer_hashes, seen_states_hashes[-1])
            for h in seen_states_hashes[:-1]:
                ans &= ~isin_via_searchsorted(current_layer_hashes, h)
            return ans

        # Applies the same mask to states and hashes.
        # If states and hashes are the same thing, it will not create a copy.
        def _apply_mask(states, hashes, mask):
            new_states = states[mask]
            new_hashes = self.hasher.hash_states(new_states) if self.hasher.is_identity else hashes[mask]
            return new_states, new_hashes

        # BFS iteration: layer2 := neighbors(layer1)-layer0-layer1.
        for i in range(1, max_diameter + 1):
            # TPU optimization: shard large layers across cores
            if enable_tpu_sharding and self.is_tpu:
                layer1 = self._shard_array_for_tpu(layer1)
                layer1 = self._optimize_memory_layout_for_tpu(layer1)

            if do_batching and len(layer1) > self.batch_size:
                num_batches = int(math.ceil(layer1_hashes.shape[0] / self.batch_size))
                layer2_batches = []  # type: list[jnp.ndarray]
                layer2_hashes_batches = []  # type: list[jnp.ndarray]

                # TPU optimization: use compiled processing for batches
                for layer1_batch in tensor_split(layer1, num_batches, axis=0):
                    if self.is_tpu and len(layer1_batch) > 100:  # Use compiled version for larger batches
                        try:
                            layer2_batch, layer2_hashes_batch = self._bfs_layer_processing_compiled(
                                layer1_batch, self.hasher.hash_states(layer1_batch)
                            )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            if self.verbose > 0:
                                print(f"Compiled processing failed, using fallback: {e}")
                            layer2_batch = self.get_neighbors(layer1_batch)
                            layer2_batch, layer2_hashes_batch = self._get_unique_states(layer2_batch)
                    else:
                        layer2_batch = self.get_neighbors(layer1_batch)
                        layer2_batch, layer2_hashes_batch = self._get_unique_states(layer2_batch)

                    mask = _remove_seen_states(layer2_hashes_batch)
                    for other_batch_hashes in layer2_hashes_batches:
                        mask &= ~isin_via_searchsorted(layer2_hashes_batch, other_batch_hashes)
                    layer2_batch, layer2_hashes_batch = _apply_mask(layer2_batch, layer2_hashes_batch, mask)
                    layer2_batches.append(layer2_batch)
                    layer2_hashes_batches.append(layer2_hashes_batch)

                layer2_hashes = concatenate_arrays(layer2_hashes_batches)
                layer2_hashes, _ = sort_with_indices(layer2_hashes, stable=True)
                layer2 = (
                    layer2_hashes.reshape((-1, 1)) if self.hasher.is_identity else concatenate_arrays(layer2_batches)
                )
            else:
                # TPU optimization: use compiled processing for single large layers
                if self.is_tpu and len(layer1) > 100:
                    try:
                        layer2, layer2_hashes = self._bfs_layer_processing_compiled(layer1, layer1_hashes)
                        if return_all_edges:
                            edges_list_starts += [jnp.repeat(layer1_hashes, self.definition.n_generators)]
                            edges_list_ends.append(layer2_hashes)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        if self.verbose > 0:
                            print(f"Compiled processing failed, using fallback: {e}")
                        layer1_neighbors = self.get_neighbors(layer1)
                        layer1_neighbors_hashes = self.hasher.hash_states(layer1_neighbors)
                        if return_all_edges:
                            edges_list_starts += [jnp.repeat(layer1_hashes, self.definition.n_generators)]
                            edges_list_ends.append(layer1_neighbors_hashes)
                        layer2, layer2_hashes = self._get_unique_states(
                            layer1_neighbors, hashes=layer1_neighbors_hashes
                        )
                else:
                    layer1_neighbors = self.get_neighbors(layer1)
                    layer1_neighbors_hashes = self.hasher.hash_states(layer1_neighbors)
                    if return_all_edges:
                        edges_list_starts += [jnp.repeat(layer1_hashes, self.definition.n_generators)]
                        edges_list_ends.append(layer1_neighbors_hashes)
                    layer2, layer2_hashes = self._get_unique_states(layer1_neighbors, hashes=layer1_neighbors_hashes)

                mask = _remove_seen_states(layer2_hashes)
                layer2, layer2_hashes = _apply_mask(layer2, layer2_hashes, mask)

            if layer2.shape[0] * layer2.shape[1] * 8 > 0.1 * self.memory_limit_bytes:
                self.free_memory()
            if return_all_hashes:
                all_layers_hashes.append(layer1_hashes)
            if len(layer2) == 0:
                full_graph_explored = True
                break
            if self.verbose >= 2:
                print(f"Layer {i}: {len(layer2)} states.")
            layer_sizes.append(len(layer2))
            if len(layer2) <= max_layer_size_to_store:
                layers[i] = self.decode_states(layer2)

            layer1 = layer2
            layer1_hashes = layer2_hashes
            seen_states_hashes.append(layer2_hashes)
            if self.definition.generators_inverse_closed:
                # Only keep hashes for last 2 layers.
                seen_states_hashes = seen_states_hashes[-2:]
            if len(layer2) >= max_layer_size_to_explore:
                break

        if return_all_hashes and not full_graph_explored:
            all_layers_hashes.append(layer1_hashes)

        if not full_graph_explored and self.verbose > 0:
            print("BFS stopped before graph was fully explored.")

        edges_list_hashes: Optional[jnp.ndarray] = None
        if return_all_edges:
            if not full_graph_explored:
                # Add copy of edges between last 2 layers, but in opposite direction.
                # This is done so adjacency matrix is symmetric.
                v1, v2 = edges_list_starts[-1], edges_list_ends[-1]
                edges_list_starts.append(v2)
                edges_list_ends.append(v1)
            edges_list_hashes = stack_arrays(
                [concatenate_arrays(edges_list_starts), concatenate_arrays(edges_list_ends)]
            ).T
        vertices_hashes: Optional[jnp.ndarray] = None
        if return_all_hashes:
            vertices_hashes = concatenate_arrays(all_layers_hashes)

        # Always store the last layer.
        last_layer_id = len(layer_sizes) - 1
        if full_graph_explored and last_layer_id not in layers:
            layers[last_layer_id] = self.decode_states(layer1)

        # Convert JAX arrays to PyTorch tensors for BfsResult compatibility
        # Use the globally imported torch module

        def jax_to_torch(jax_array):
            """Convert JAX array to PyTorch tensor."""
            if jax_array is None:
                return None
            if isinstance(jax_array, jnp.ndarray):
                # Convert JAX array to numpy, then to PyTorch
                # Make a copy to ensure the array is writable
                numpy_array = np.array(jax_array)
                return torch.from_numpy(numpy_array)
            return jax_array

        # Convert layers to PyTorch tensors
        torch_layers = {}
        for layer_id, layer_states in layers.items():
            torch_layers[layer_id] = jax_to_torch(layer_states)

        # Convert vertices_hashes to PyTorch tensor if present
        torch_vertices_hashes = jax_to_torch(vertices_hashes)

        # Convert edges_list_hashes to PyTorch tensor if present
        torch_edges_list_hashes = jax_to_torch(edges_list_hashes)

        return BfsResult(
            layer_sizes=layer_sizes,
            layers=torch_layers,
            bfs_completed=full_graph_explored,
            vertices_hashes=torch_vertices_hashes,
            edges_list_hashes=torch_edges_list_hashes,
            graph=self.definition,
        )

    def random_walks(
        self,
        *,
        width=5,
        length=10,
        mode="classic",
        start_state: Union[None, jnp.ndarray, np.ndarray, list] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random walks on this graph.

        The following modes of random walk generation are supported:

          * "classic" - random walk is a path in this graph starting from `start_state`, where on each step the next
            edge is chosen randomly with equal probability. We generate `width` such random walks independently.
            The output will have exactly ``width*length`` states.
            i-th random walk can be extracted as: ``[x[i+j*width] for j in range(length)]``.
            ``y[i]`` is equal to number of random steps it took to get to state ``x[i]``.
            Note that in this mode a lot of states will have overestimated distance (meaning ``y[i]`` may be larger than
            the length of the shortest path from ``x[i]`` to `start_state`).
            The same state may appear multiple times with different distance in ``y``.
          * "bfs" - we perform Breadth First Search starting from ``start_state`` with one modification: if size of
            next layer is larger than ``width``, only ``width`` states (chosen randomly) will be kept.
            We also remove states from current layer if they appeared on some previous layer (so this also can be
            called "non-backtracking random walk").
            All states in the output are unique. ``y`` still can be overestimated, but it will be closer to the true
            distance than in "classic" mode. Size of output is ``<= width*length``.
            If ``width`` and ``length`` are large enough (``width`` at least as large as largest BFS layer, and
            ``length >= diameter``), this will return all states and true distances to the start state.

        Args:
            width: Number of random walks to generate.
            length: Length of each random walk.
            start_state: State from which to start random walk. Defaults to the central state.
            mode: Type of random walk (see above). Defaults to "classic".

        Returns:
            Pair of arrays ``x, y``. ``x`` contains states. ``y[i]`` is the estimated distance from start state
            to state ``x[i]``.
        """
        start_state = self.encode_states(start_state or self.central_state)
        if mode == "classic":
            return self._random_walks_classic(width, length, start_state)
        elif mode == "bfs":
            return self._random_walks_bfs(width, length, start_state)
        else:
            raise ValueError("Unknown mode:", mode)

    def _random_walks_classic(
        self, width: int, length: int, start_state: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Classic random walks implementation."""
        # Allocate memory.
        x_shape = (width * length, self.encoded_state_size)
        x = jnp.zeros(x_shape, dtype=jnp.int64)
        y = jnp.zeros(width * length, dtype=jnp.int32)

        # First state in each walk is the start state.
        x = x.at[:width, :].set(start_state.reshape((-1,)))
        y = y.at[:width].set(0)

        # Initialize random key
        rng_key = jrandom.PRNGKey(42)  # Use fixed seed for reproducibility

        # Main loop.
        for i_step in range(1, length):
            y = y.at[i_step * width : (i_step + 1) * width].set(i_step)
            rng_key, subkey = jrandom.split(rng_key)
            gen_idx = jrandom.randint(subkey, (width,), 0, self.definition.n_generators)
            src = x[(i_step - 1) * width : i_step * width, :]

            # Apply generators based on random selection
            dst_states = []
            for j in range(self.definition.n_generators):
                # Go to next state for walks where we chose to use j-th generator on this step.
                mask = gen_idx == j
                prev_states = src[mask, :]
                if len(prev_states) > 0:
                    next_states = self._apply_generator_batched(j, prev_states)
                    dst_states.append((mask, next_states))

            # Update x with new states
            dst = x[i_step * width : (i_step + 1) * width, :]
            for mask, next_states in dst_states:
                dst = dst.at[mask, :].set(next_states)
            x = x.at[i_step * width : (i_step + 1) * width, :].set(dst)

        return self.decode_states(x), y

    def _random_walks_bfs(self, width: int, length: int, start_state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """BFS-style random walks implementation."""
        # Use imported JAXHashSet
        if JAXHashSet is None:
            raise ImportError("JAXHashSet not available")

        x_hashes = JAXHashSet()
        x_hashes.add_sorted_hashes(self.hasher.hash_states(start_state))
        x = [start_state]
        y = [jnp.full((1,), 0, dtype=jnp.int32)]

        # Initialize random key
        rng_key = jrandom.PRNGKey(42)

        for i_step in range(1, length):
            next_states = self.get_neighbors(x[-1])
            next_states, next_states_hashes = self._get_unique_states(next_states)
            mask = x_hashes.get_mask_to_remove_seen_hashes(next_states_hashes)
            next_states, next_states_hashes = next_states[mask], next_states_hashes[mask]
            layer_size = len(next_states)
            if layer_size == 0:
                break
            if layer_size > width:
                rng_key, subkey = jrandom.split(rng_key)
                random_indices = jrandom.choice(subkey, layer_size, (width,), replace=False)
                layer_size = width
                next_states = next_states[random_indices]
                next_states_hashes = next_states_hashes[random_indices]
            x.append(next_states)
            x_hashes.add_sorted_hashes(next_states_hashes)
            y.append(jnp.full((layer_size,), i_step, dtype=jnp.int32))
        return self.decode_states(stack_arrays(x)), concatenate_arrays(y)

    def beam_search(
        self,
        *,
        start_state: Union[jnp.ndarray, np.ndarray, list],
        predictor=None,
        beam_width=1000,
        max_iterations=1000,
        return_path=False,
    ):
        """Try to find a path from start_state to central state using Beam Search algorithm.

        Args:
            start_state: State from which to start search.
            predictor: A heuristic that estimates scores for states (lower score = closer to center).
                Defaults to Hamming distance heuristic.
            beam_width: Width of the beam (how many "best" states we consider at each step).
            max_iterations: Maximum number of iterations before giving up.
            return_path: Whether to return path (consumes much more memory if True).

        Returns:
            BeamSearchResult containing found path length and (optionally) the path itself.
        """
        # Use imported modules
        if BeamSearchResult is None or Predictor is None:
            raise ImportError("BeamSearchResult or Predictor not available")

        if predictor is None:
            predictor = Predictor(self, "hamming")  # type: ignore
        start_states = self.encode_states(start_state)
        layer1, layer1_hashes = self._get_unique_states(start_states)
        all_layers_hashes = [layer1_hashes]
        debug_scores = {}  # type: dict[int, float]

        if self.central_state_hash[0] == layer1_hashes[0]:
            # Start state is the central state.
            return BeamSearchResult(True, 0, [], debug_scores, self.definition)

        for i in range(max_iterations):
            # Create states on the next layer.
            layer2, layer2_hashes = self._get_unique_states(self.get_neighbors(layer1))

            if bool(isin_via_searchsorted(self.central_state_hash, layer2_hashes)):
                # Path found.
                path = None
                if return_path:
                    path = self._restore_path(all_layers_hashes, self.central_state)
                return BeamSearchResult(True, i + 1, path, debug_scores, self.definition)

            # Pick `beam_width` states with lowest scores.
            if len(layer2) >= beam_width:
                scores = predictor(self.decode_states(layer2))
                idx = jnp.argsort(scores)[:beam_width]
                layer2 = layer2[idx, :]
                layer2_hashes = layer2_hashes[idx]
                best_score = float(scores[idx[0]])
                debug_scores[i] = best_score
                if self.verbose >= 2:
                    print(f"Iteration {i}, best score {best_score}.")

            layer1 = layer2
            layer1_hashes = layer2_hashes
            if return_path:
                all_layers_hashes.append(layer1_hashes)

        # Path not found.
        return BeamSearchResult(False, 0, None, debug_scores, self.definition)

    def _restore_path(self, hashes: List[jnp.ndarray], to_state: Union[jnp.ndarray, np.ndarray, list]) -> List[int]:
        """Restore path from layers hashes.

        Layers must be such that there is edge from state on previous layer to state on next layer.
        First layer in `hashes` must have exactly one state, this is the start of the path.
        The end of the path is to_state.
        Last layer in `hashes` must contain a state from which there is a transition to `to_state`.
        `to_state` must be in "decoded" format.
        Length of returned path is equal to number of layers.
        """
        inv_graph = JAXCayleyGraph(self.definition.with_inverted_generators())
        assert len(hashes[0]) == 1
        path = []  # type: List[int]
        cur_state = self.decode_states(self.encode_states(to_state))

        for i in range(len(hashes) - 1, -1, -1):
            # Find hash in hashes[i] from which we could go to cur_state.
            # Corresponding state will be new_cur_state.
            # The generator index in inv_graph that moves cur_state->new_cur_state is the same as generator index
            # in this graph that moves new_cur_state->cur_state - this is what we append to the answer.
            candidates = inv_graph.get_neighbors_decoded(cur_state)
            candidates_hashes = self.hasher.hash_states(self.encode_states(candidates))
            mask = isin_via_searchsorted(candidates_hashes, hashes[i])
            assert jnp.any(mask), "Not found any neighbor on previous layer."
            gen_id = int(jnp.nonzero(mask)[0][0])
            path.append(gen_id)
            cur_state = candidates[gen_id : gen_id + 1, :]
        return path[::-1]

    def find_path_to(self, end_state: Union[jnp.ndarray, np.ndarray, list], bfs_result) -> Optional[List[int]]:
        """Find path from central_state to end_state using pre-computed BfsResult.

        Args:
            end_state: Final state of the path.
            bfs_result: Pre-computed BFS result (call `bfs(return_all_hashes=True)` to get this).

        Returns:
            The found path (list of generator ids), or None if `end_state` is not reachable from `start_state`.
        """
        end_state_hash = self.hasher.hash_states(self.encode_states(end_state))
        assert bfs_result.vertices_hashes is not None, "Run bfs with return_all_hashes=True."
        i = 0
        layers_hashes = []  # type: List[jnp.ndarray]
        for layer_size in bfs_result.layer_sizes:
            cur_layer = bfs_result.vertices_hashes[i : i + layer_size]
            i += layer_size
            if bool(isin_via_searchsorted(end_state_hash, cur_layer)):
                return self._restore_path(layers_hashes, end_state)
            layers_hashes.append(cur_layer)
        return None

    def find_path_from(self, start_state: Union[jnp.ndarray, np.ndarray, list], bfs_result) -> Optional[List[int]]:
        """Find path from start_state to central_state using pre-computed BfsResult.

        This is possible only for inverse-closed generators.

        Args:
            start_state: First state of the path.
            bfs_result: Pre-computed BFS result (call `bfs(return_all_hashes=True)` to get this).

        Returns:
            The found path (list of generator ids), or None if central_state is not reachable from start_state.
        """
        assert self.definition.generators_inverse_closed
        path_to = self.find_path_to(start_state, bfs_result)
        if path_to is None:
            return None
        return self.definition.revert_path(path_to)

    def to_networkx_graph(self):
        """Convert to NetworkX graph."""
        return self.bfs(
            max_layer_size_to_store=10**18, return_all_edges=True, return_all_hashes=True
        ).to_networkx_graph()

    def _should_shard_array(self, array: jnp.ndarray) -> bool:
        """Determine if array should be sharded across TPU cores."""
        if not self.is_tpu or self.num_devices <= 1:
            return False

        # Shard if array is large enough and has sufficient batch dimension
        total_elements = array.size
        batch_size = array.shape[0] if array.ndim > 0 else 1

        return total_elements > self.tpu_shard_threshold and batch_size >= self.num_devices

    def _shard_array_for_tpu(self, array: jnp.ndarray) -> jnp.ndarray:
        """Shard array across TPU cores for parallel processing."""
        if not self._should_shard_array(array):
            return array

        try:
            # Use JAX's automatic sharding
            if PositionalSharding is None:
                raise ImportError("PositionalSharding not available")

            # Create sharding specification
            sharding = PositionalSharding(self.device_manager.devices).reshape(self.num_devices, 1)

            # Shard along the batch dimension (axis 0)
            return jax.device_put(array, sharding)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if self.verbose > 0:
                print(f"TPU sharding failed, using single device: {e}")
            return array

    @partial(jit, static_argnums=(0,))
    def _bfs_layer_processing_compiled(
        self, layer_states: jnp.ndarray, _layer_hashes: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """JIT-compiled BFS layer processing for TPU optimization."""
        # Get neighbors for all states in the layer
        neighbors = self.get_neighbors(layer_states)

        # Hash the neighbors
        neighbor_hashes = self.hasher.hash_states(neighbors)

        # Remove duplicates within this batch
        unique_neighbors, unique_hashes = self._get_unique_states(neighbors, neighbor_hashes)

        return unique_neighbors, unique_hashes

    def _optimize_memory_layout_for_tpu(self, array: jnp.ndarray) -> jnp.ndarray:
        """Optimize memory layout for TPU access patterns."""
        if not self.is_tpu:
            return array

        # Ensure arrays are contiguous and properly aligned for TPU
        # TPUs prefer certain memory layouts for optimal performance
        if array.ndim >= 2:
            # JAX arrays are typically already in optimal layout
            # No need for explicit contiguous array conversion in JAX
            pass

        return array

    def free_memory(self):
        """Free memory and clear caches."""
        if self.verbose >= 1:
            print("Freeing memory...")

        # Clear JAX compilation cache
        self.device_manager.clear_cache()

        # Python garbage collection
        gc.collect()

    @property
    def generators(self):
        """Generators of this Cayley graph."""
        return self.definition.generators

    def __str__(self) -> str:
        """String representation."""
        return f"JAXCayleyGraph(device={self.device_manager.device_type}, generators={self.definition.n_generators})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"JAXCayleyGraph(definition={self.definition}, "
            f"device={self.device_manager.device_type}, "
            f"encoded_size={self.encoded_state_size})"
        )
