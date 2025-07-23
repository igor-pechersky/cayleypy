"""JAX-based hash set implementation for efficient state tracking.

This module provides a JAX equivalent of the TorchHashSet class,
optimized for TPU/GPU computation with vectorized operations.
"""

from typing import List
import warnings

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from .jax_tensor_ops import isin_via_searchsorted, sort_with_indices, concatenate_arrays


class JAXHashSet:
    """A set of int64 numbers, backed by one or more sorted JAX arrays.

    This is a JAX equivalent of TorchHashSet, optimized for TPU/GPU computation
    with automatic consolidation and vectorized operations.
    """

    def __init__(self, max_segments: int = 10):
        """Initialize empty hash set.

        Args:
            max_segments: Maximum number of segments before consolidation
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Install with: pip install jax[tpu] or pip install jax[cuda]")

        self.data: List[jnp.ndarray] = []
        self.max_segments = max_segments

    def add_sorted_hashes(self, sorted_numbers: jnp.ndarray) -> None:
        """Add sorted hash values to the set.

        IMPORTANT: Assumes that new numbers are sorted and do not appear in the set before.
        This is a direct port of the PyTorch version for compatibility.

        Args:
            sorted_numbers: Sorted array of hash values to add
        """
        if len(sorted_numbers) == 0:
            return

        self.data.append(sorted_numbers)

        # Consolidate segments if we have too many
        if len(self.data) >= self.max_segments:
            self._consolidate_segments()

    @jit
    def _consolidate_segments(self) -> None:
        """Consolidate all segments into a single sorted array."""
        if len(self.data) <= 1:
            return

        # Concatenate all segments
        all_data = concatenate_arrays(self.data, axis=0)

        # Sort the combined data
        sorted_data, _ = sort_with_indices(all_data)

        # Replace all segments with single consolidated segment
        self.data = [sorted_data]

    def get_mask_to_remove_seen_hashes(self, x: jnp.ndarray) -> jnp.ndarray:
        """Get boolean mask indicating which elements in x are NOT in the set.

        This is used to filter out previously seen hash values.

        Args:
            x: Array of hash values to check

        Returns:
            Boolean mask where True indicates unseen (should keep) elements
        """
        if len(self.data) == 0:
            # No hashes stored yet, keep all elements
            return jnp.ones_like(x, dtype=bool)

        # Start with mask that keeps all elements
        mask = jnp.ones_like(x, dtype=bool)

        # Check against each segment
        for segment in self.data:
            if len(segment) > 0:
                # Elements that are in this segment should be removed
                in_segment = isin_via_searchsorted(x, segment)
                mask = jnp.logical_and(mask, jnp.logical_not(in_segment))

        return mask

    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        """Check if elements are contained in the set.

        Args:
            x: Array of values to check

        Returns:
            Boolean array indicating membership
        """
        if len(self.data) == 0:
            return jnp.zeros_like(x, dtype=bool)

        # Check against each segment
        result = jnp.zeros_like(x, dtype=bool)
        for segment in self.data:
            if len(segment) > 0:
                in_segment = isin_via_searchsorted(x, segment)
                result = jnp.logical_or(result, in_segment)

        return result

    def size(self) -> int:
        """Get total number of unique elements in the set.

        Returns:
            Number of unique elements
        """
        if len(self.data) == 0:
            return 0

        # If we have multiple segments, we need to consolidate to get accurate count
        if len(self.data) > 1:
            self._consolidate_segments()

        return len(self.data[0]) if self.data else 0

    def is_empty(self) -> bool:
        """Check if the set is empty.

        Returns:
            True if set is empty
        """
        return len(self.data) == 0 or all(len(segment) == 0 for segment in self.data)

    def clear(self) -> None:
        """Clear all elements from the set."""
        self.data = []

    def get_all_elements(self) -> jnp.ndarray:
        """Get all elements in the set as a sorted array.

        Returns:
            Sorted array of all elements
        """
        if len(self.data) == 0:
            return jnp.array([], dtype=jnp.int64)

        # Consolidate if needed
        if len(self.data) > 1:
            self._consolidate_segments()

        return self.data[0] if self.data else jnp.array([], dtype=jnp.int64)

    def union(self, other: "JAXHashSet") -> "JAXHashSet":
        """Create union with another hash set.

        Args:
            other: Another JAXHashSet

        Returns:
            New JAXHashSet containing union of both sets
        """
        result = JAXHashSet(max_segments=self.max_segments)

        # Add all elements from both sets
        for segment in self.data:
            if len(segment) > 0:
                result.add_sorted_hashes(segment)

        for segment in other.data:
            if len(segment) > 0:
                # Need to check for duplicates when adding from other set
                mask = result.get_mask_to_remove_seen_hashes(segment)
                new_elements = segment[mask]
                if len(new_elements) > 0:
                    # Sort the new elements before adding
                    sorted_new, _ = sort_with_indices(new_elements)
                    result.add_sorted_hashes(sorted_new)

        return result

    def intersection(self, other: "JAXHashSet") -> "JAXHashSet":
        """Create intersection with another hash set.

        Args:
            other: Another JAXHashSet

        Returns:
            New JAXHashSet containing intersection of both sets
        """
        result = JAXHashSet(max_segments=self.max_segments)

        if self.is_empty() or other.is_empty():
            return result

        # Get all elements from both sets
        self_elements = self.get_all_elements()
        other_elements = other.get_all_elements()

        # Find intersection using isin
        in_other = isin_via_searchsorted(self_elements, other_elements)
        intersection_elements = self_elements[in_other]

        if len(intersection_elements) > 0:
            result.add_sorted_hashes(intersection_elements)

        return result

    def difference(self, other: "JAXHashSet") -> "JAXHashSet":
        """Create difference with another hash set (elements in self but not in other).

        Args:
            other: Another JAXHashSet

        Returns:
            New JAXHashSet containing difference
        """
        result = JAXHashSet(max_segments=self.max_segments)

        if self.is_empty():
            return result

        if other.is_empty():
            # If other is empty, difference is just a copy of self
            for segment in self.data:
                if len(segment) > 0:
                    result.add_sorted_hashes(segment)
            return result

        # Get all elements from both sets
        self_elements = self.get_all_elements()
        other_elements = other.get_all_elements()

        # Find elements in self but not in other
        not_in_other = jnp.logical_not(isin_via_searchsorted(self_elements, other_elements))
        difference_elements = self_elements[not_in_other]

        if len(difference_elements) > 0:
            result.add_sorted_hashes(difference_elements)

        return result

    def __len__(self) -> int:
        """Get number of elements in the set."""
        return self.size()

    def __bool__(self) -> bool:
        """Check if set is non-empty."""
        return not self.is_empty()

    def __str__(self) -> str:
        """String representation of the set."""
        size = self.size()
        segments = len(self.data)
        return f"JAXHashSet(size={size}, segments={segments})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"JAXHashSet(size={self.size()}, segments={len(self.data)}, max_segments={self.max_segments})"


# Utility functions for working with JAXHashSet
def create_hash_set_from_array(array: jnp.ndarray, max_segments: int = 10) -> JAXHashSet:
    """Create a JAXHashSet from an array of values.

    Args:
        array: Array of values to add to set
        max_segments: Maximum segments before consolidation

    Returns:
        JAXHashSet containing unique sorted values
    """
    hash_set = JAXHashSet(max_segments=max_segments)

    if len(array) > 0:
        # Sort and get unique values
        sorted_array, _ = sort_with_indices(array.flatten())

        # Remove duplicates by comparing adjacent elements
        if len(sorted_array) > 1:
            unique_mask = jnp.concatenate([jnp.array([True]), sorted_array[1:] != sorted_array[:-1]])
            unique_values = sorted_array[unique_mask]
        else:
            unique_values = sorted_array

        hash_set.add_sorted_hashes(unique_values)

    return hash_set


def merge_hash_sets(hash_sets: List[JAXHashSet]) -> JAXHashSet:
    """Merge multiple hash sets into one.

    Args:
        hash_sets: List of JAXHashSet objects to merge

    Returns:
        New JAXHashSet containing union of all input sets
    """
    if not hash_sets:
        return JAXHashSet()

    result = JAXHashSet()

    for hash_set in hash_sets:
        result = result.union(hash_set)

    return result
