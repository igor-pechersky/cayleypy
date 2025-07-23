"""Unit tests for JAX hash set implementation."""

from unittest.mock import patch

import pytest

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from cayleypy.jax_hash_set import JAXHashSet, create_hash_set_from_array, merge_hash_sets, JAX_AVAILABLE


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXHashSet:
    """Test cases for JAXHashSet."""

    def test_empty_hash_set(self):
        """Test empty hash set behavior."""
        hash_set = JAXHashSet()

        assert hash_set.is_empty()
        assert len(hash_set) == 0
        assert hash_set.size() == 0
        assert not bool(hash_set)

    def test_add_sorted_hashes(self):
        """Test adding sorted hashes."""
        hash_set = JAXHashSet()

        # Add first batch
        hashes1 = jnp.array([1, 3, 5, 7])
        hash_set.add_sorted_hashes(hashes1)

        assert not hash_set.is_empty()
        assert len(hash_set) == 4
        assert bool(hash_set)

    def test_get_mask_to_remove_seen_hashes(self):
        """Test mask generation for unseen hashes."""
        hash_set = JAXHashSet()

        # Add some hashes
        hash_set.add_sorted_hashes(jnp.array([2, 4, 6]))

        # Test with mixed seen/unseen hashes
        test_hashes = jnp.array([1, 2, 3, 4, 5, 6, 7])
        mask = hash_set.get_mask_to_remove_seen_hashes(test_hashes)

        # Should keep 1, 3, 5, 7 (unseen) and remove 2, 4, 6 (seen)
        expected_mask = jnp.array([True, False, True, False, True, False, True])
        assert jnp.array_equal(mask, expected_mask)

    def test_contains(self):
        """Test membership testing."""
        hash_set = JAXHashSet()
        hash_set.add_sorted_hashes(jnp.array([10, 20, 30]))

        test_values = jnp.array([5, 10, 15, 20, 25, 30, 35])
        result = hash_set.contains(test_values)

        expected = jnp.array([False, True, False, True, False, True, False])
        assert jnp.array_equal(result, expected)

    def test_consolidation(self):
        """Test automatic consolidation of segments."""
        hash_set = JAXHashSet(max_segments=3)

        # Add multiple segments to trigger consolidation
        hash_set.add_sorted_hashes(jnp.array([1, 3]))
        hash_set.add_sorted_hashes(jnp.array([2, 4]))
        hash_set.add_sorted_hashes(jnp.array([5, 7]))

        # Should have 3 segments now
        assert len(hash_set.data) == 3

        # Add one more to trigger consolidation
        hash_set.add_sorted_hashes(jnp.array([6, 8]))

        # Should be consolidated to 1 segment
        assert len(hash_set.data) == 1

        # All elements should still be accessible
        all_elements = hash_set.get_all_elements()
        expected = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
        assert jnp.array_equal(all_elements, expected)

    def test_get_all_elements(self):
        """Test getting all elements as sorted array."""
        hash_set = JAXHashSet()

        # Add elements in different batches
        hash_set.add_sorted_hashes(jnp.array([5, 10]))
        hash_set.add_sorted_hashes(jnp.array([1, 15]))
        hash_set.add_sorted_hashes(jnp.array([3, 7]))

        all_elements = hash_set.get_all_elements()
        expected = jnp.array([1, 3, 5, 7, 10, 15])
        assert jnp.array_equal(all_elements, expected)

    def test_clear(self):
        """Test clearing the hash set."""
        hash_set = JAXHashSet()
        hash_set.add_sorted_hashes(jnp.array([1, 2, 3]))

        assert not hash_set.is_empty()

        hash_set.clear()

        assert hash_set.is_empty()
        assert len(hash_set) == 0

    def test_union(self):
        """Test union of two hash sets."""
        hash_set1 = JAXHashSet()
        hash_set1.add_sorted_hashes(jnp.array([1, 3, 5]))

        hash_set2 = JAXHashSet()
        hash_set2.add_sorted_hashes(jnp.array([2, 4, 6]))

        union_set = hash_set1.union(hash_set2)

        all_elements = union_set.get_all_elements()
        expected = jnp.array([1, 2, 3, 4, 5, 6])
        assert jnp.array_equal(all_elements, expected)

    def test_union_with_overlap(self):
        """Test union with overlapping elements."""
        hash_set1 = JAXHashSet()
        hash_set1.add_sorted_hashes(jnp.array([1, 2, 3]))

        hash_set2 = JAXHashSet()
        hash_set2.add_sorted_hashes(jnp.array([2, 3, 4]))

        union_set = hash_set1.union(hash_set2)

        all_elements = union_set.get_all_elements()
        expected = jnp.array([1, 2, 3, 4])
        assert jnp.array_equal(all_elements, expected)

    def test_intersection(self):
        """Test intersection of two hash sets."""
        hash_set1 = JAXHashSet()
        hash_set1.add_sorted_hashes(jnp.array([1, 2, 3, 4]))

        hash_set2 = JAXHashSet()
        hash_set2.add_sorted_hashes(jnp.array([3, 4, 5, 6]))

        intersection_set = hash_set1.intersection(hash_set2)

        all_elements = intersection_set.get_all_elements()
        expected = jnp.array([3, 4])
        assert jnp.array_equal(all_elements, expected)

    def test_intersection_empty(self):
        """Test intersection with no common elements."""
        hash_set1 = JAXHashSet()
        hash_set1.add_sorted_hashes(jnp.array([1, 2, 3]))

        hash_set2 = JAXHashSet()
        hash_set2.add_sorted_hashes(jnp.array([4, 5, 6]))

        intersection_set = hash_set1.intersection(hash_set2)

        assert intersection_set.is_empty()

    def test_difference(self):
        """Test difference of two hash sets."""
        hash_set1 = JAXHashSet()
        hash_set1.add_sorted_hashes(jnp.array([1, 2, 3, 4, 5]))

        hash_set2 = JAXHashSet()
        hash_set2.add_sorted_hashes(jnp.array([3, 4, 6, 7]))

        difference_set = hash_set1.difference(hash_set2)

        all_elements = difference_set.get_all_elements()
        expected = jnp.array([1, 2, 5])
        assert jnp.array_equal(all_elements, expected)

    def test_difference_empty_other(self):
        """Test difference with empty other set."""
        hash_set1 = JAXHashSet()
        hash_set1.add_sorted_hashes(jnp.array([1, 2, 3]))

        hash_set2 = JAXHashSet()  # Empty

        difference_set = hash_set1.difference(hash_set2)

        all_elements = difference_set.get_all_elements()
        expected = jnp.array([1, 2, 3])
        assert jnp.array_equal(all_elements, expected)

    def test_string_representations(self):
        """Test string representation methods."""
        hash_set = JAXHashSet(max_segments=5)
        hash_set.add_sorted_hashes(jnp.array([1, 2, 3]))

        str_repr = str(hash_set)
        assert "JAXHashSet" in str_repr
        assert "size=3" in str_repr

        repr_str = repr(hash_set)
        assert "JAXHashSet" in repr_str
        assert "max_segments=5" in repr_str

    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        hash_set = JAXHashSet()

        # Adding empty array should not change state
        hash_set.add_sorted_hashes(jnp.array([]))
        assert hash_set.is_empty()

        # Mask for empty test should return all True
        mask = hash_set.get_mask_to_remove_seen_hashes(jnp.array([1, 2, 3]))
        expected = jnp.array([True, True, True])
        assert jnp.array_equal(mask, expected)

    def test_large_hash_set(self):
        """Test hash set with large number of elements."""
        hash_set = JAXHashSet()

        # Add large sorted array
        large_hashes = jnp.arange(10000)
        hash_set.add_sorted_hashes(large_hashes)

        assert hash_set.size() == 10000

        # Test membership
        test_values = jnp.array([100, 5000, 9999, 10000])
        result = hash_set.contains(test_values)
        expected = jnp.array([True, True, True, False])
        assert jnp.array_equal(result, expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXHashSetUtilities:
    """Test utility functions for JAXHashSet."""

    def test_create_hash_set_from_array(self):
        """Test creating hash set from array."""
        # Array with duplicates
        array = jnp.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
        hash_set = create_hash_set_from_array(array)

        all_elements = hash_set.get_all_elements()
        expected = jnp.array([1, 2, 3, 4, 5, 6, 9])
        assert jnp.array_equal(all_elements, expected)

    def test_create_hash_set_from_empty_array(self):
        """Test creating hash set from empty array."""
        array = jnp.array([])
        hash_set = create_hash_set_from_array(array)

        assert hash_set.is_empty()

    def test_merge_hash_sets(self):
        """Test merging multiple hash sets."""
        hash_set1 = JAXHashSet()
        hash_set1.add_sorted_hashes(jnp.array([1, 3, 5]))

        hash_set2 = JAXHashSet()
        hash_set2.add_sorted_hashes(jnp.array([2, 4, 6]))

        hash_set3 = JAXHashSet()
        hash_set3.add_sorted_hashes(jnp.array([5, 7, 9]))

        merged = merge_hash_sets([hash_set1, hash_set2, hash_set3])

        all_elements = merged.get_all_elements()
        expected = jnp.array([1, 2, 3, 4, 5, 6, 7, 9])
        assert jnp.array_equal(all_elements, expected)

    def test_merge_empty_list(self):
        """Test merging empty list of hash sets."""
        merged = merge_hash_sets([])
        assert merged.is_empty()

    def test_merge_single_hash_set(self):
        """Test merging single hash set."""
        hash_set = JAXHashSet()
        hash_set.add_sorted_hashes(jnp.array([1, 2, 3]))

        merged = merge_hash_sets([hash_set])

        all_elements = merged.get_all_elements()
        expected = jnp.array([1, 2, 3])
        assert jnp.array_equal(all_elements, expected)


class TestJAXHashSetNotAvailable:
    """Test behavior when JAX is not available."""

    def test_hash_set_without_jax(self):
        """Test that JAXHashSet raises ImportError when JAX not available."""
        with patch("cayleypy.jax_hash_set.JAX_AVAILABLE", False):
            with pytest.raises(ImportError, match="JAX is not available"):
                JAXHashSet()


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXHashSetPerformance:
    """Performance tests for JAXHashSet."""

    def test_large_scale_operations(self):
        """Test hash set with large scale operations."""
        hash_set = JAXHashSet()

        # Add multiple large batches
        for i in range(10):
            batch = jnp.arange(i * 1000, (i + 1) * 1000)
            hash_set.add_sorted_hashes(batch)

        assert hash_set.size() == 10000

        # Test large membership query
        test_values = jnp.arange(0, 15000, 100)  # Every 100th value
        result = hash_set.contains(test_values)

        # Values 0-9999 should be in set, 10000+ should not
        expected = test_values < 10000
        assert jnp.array_equal(result, expected)

    def test_consolidation_performance(self):
        """Test performance of consolidation with many segments."""
        hash_set = JAXHashSet(max_segments=20)

        # Add many small segments to test consolidation
        for i in range(25):  # More than max_segments
            batch = jnp.array([i * 10, i * 10 + 1])
            hash_set.add_sorted_hashes(batch)

        # Should be consolidated
        assert len(hash_set.data) == 1
        assert hash_set.size() == 50

    def test_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        hash_set = JAXHashSet()

        # Add very large sorted array
        large_array = jnp.arange(100000)
        hash_set.add_sorted_hashes(large_array)

        # Test efficient membership queries
        test_queries = jnp.array([0, 50000, 99999, 100000, 200000])
        result = hash_set.contains(test_queries)
        expected = jnp.array([True, True, True, False, False])

        assert jnp.array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
