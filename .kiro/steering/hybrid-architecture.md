---
inclusion: fileMatch
fileMatchPattern: '*hybrid*|*bfs*|*beam*|*hash*'
---

# Hybrid Device Architecture Guidelines for CayleyPy

## Core Principles

The hybrid architecture addresses TPU int64 limitations by intelligently orchestrating operations across multiple devices (TPU, GPU, CPU) based on their strengths and constraints.

## Device Specialization Strategy

### TPU Strengths
- Massive parallel tensor operations
- Neural network inference and training
- Large-scale matrix multiplications
- Vectorized computations with int32 data

### TPU Limitations
- No int64 operations support
- Limited dynamic memory allocation
- Static tensor shapes required
- No native hash tables or sets

### CPU Strengths
- Full int64 support for precise hashing
- Dynamic data structures (hash tables, sets)
- Complex control flow and logic
- Flexible memory management

### GPU Strengths
- Good balance of parallelism and flexibility
- Support for both int32 and int64 operations
- Dynamic memory allocation
- Fallback option when TPU/CPU specialization isn't optimal

## Implementation Patterns

### Device Selection Logic
```python
def choose_optimal_device(operation_type: str, data_characteristics: Dict) -> str:
    """Choose optimal device based on operation and data characteristics."""
    
    if operation_type == "hash_computation":
        if data_characteristics["requires_int64"] or data_characteristics["size"] > 10000:
            return "cpu"  # CPU for precise int64 hashing
        else:
            return "tpu"  # TPU for fast int32 approximate hashing
    
    elif operation_type == "tensor_operations":
        if data_characteristics["static_shape"] and data_characteristics["size"] > 1000:
            return "tpu"  # TPU for large static tensor operations
        else:
            return "gpu"  # GPU for dynamic or smaller operations
    
    elif operation_type == "neural_network":
        return "tpu"  # TPU optimal for neural network operations
    
    elif operation_type == "deduplication":
        if data_characteristics["approximate_ok"]:
            return "tpu"  # TPU for quick approximate deduplication
        else:
            return "cpu"  # CPU for precise deduplication
    
    else:
        return "cpu"  # Default fallback
```

### State Encoding for Multi-Device
```python
class MultiDeviceStateEncoder:
    """Encode states appropriately for different devices."""
    
    def encode_for_device(self, states: jnp.ndarray, target_device: str) -> jnp.ndarray:
        """Encode states for specific device capabilities."""
        
        if target_device == "tpu":
            # TPU requires int32 and static shapes
            if self.state_size <= 2**15:
                return states.astype(jnp.int32)
            else:
                return self._chunk_encode_int32(states)
        
        elif target_device == "cpu":
            # CPU can handle full int64 precision
            return states.astype(jnp.int64)
        
        elif target_device == "gpu":
            # GPU flexible - choose based on precision needs
            if self.requires_high_precision:
                return states.astype(jnp.int64)
            else:
                return states.astype(jnp.int32)
        
        else:
            return states  # No encoding needed
```

### Hierarchical Processing Pipeline
```python
class HierarchicalProcessor:
    """Multi-stage processing with device specialization."""
    
    def process_large_batch(self, data: jnp.ndarray) -> jnp.ndarray:
        """Process large batches using hierarchical approach."""
        
        # Stage 1: TPU quick filtering (removes obvious duplicates/irrelevant data)
        tpu_encoded = self.encoder.encode_for_device(data, "tpu")
        quick_filtered = self.tpu_processor.quick_filter(tpu_encoded)
        
        # Stage 2: GPU intermediate processing (if needed)
        if len(quick_filtered) > self.gpu_threshold:
            gpu_processed = self.gpu_processor.intermediate_process(quick_filtered)
        else:
            gpu_processed = quick_filtered
        
        # Stage 3: CPU precise processing (final accurate results)
        cpu_encoded = self.encoder.encode_for_device(gpu_processed, "cpu")
        final_result = self.cpu_processor.precise_process(cpu_encoded)
        
        return final_result
```

### Data Transfer Optimization
```python
class DataTransferOptimizer:
    """Optimize data transfers between devices."""
    
    def transfer_with_batching(self, data: jnp.ndarray, 
                              source_device: str, target_device: str) -> jnp.ndarray:
        """Transfer data between devices with optimal batching."""
        
        # Calculate optimal batch size based on device memory and bandwidth
        batch_size = self._calculate_optimal_batch_size(
            data.shape, source_device, target_device
        )
        
        if len(data) <= batch_size:
            # Small data - direct transfer
            return jax.device_put(data, self.get_device(target_device))
        
        else:
            # Large data - batched transfer with overlap
            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                transferred_batch = jax.device_put(batch, self.get_device(target_device))
                results.append(transferred_batch)
            
            return jnp.concatenate(results, axis=0)
```

### Error Handling and Fallback
```python
class HybridErrorHandler:
    """Handle device-specific errors with intelligent fallback."""
    
    def execute_with_fallback(self, operation: callable, data: jnp.ndarray, 
                             preferred_device: str) -> jnp.ndarray:
        """Execute operation with automatic fallback on device errors."""
        
        device_priority = self._get_fallback_priority(preferred_device)
        
        for device in device_priority:
            try:
                # Encode data for target device
                encoded_data = self.encoder.encode_for_device(data, device)
                
                # Execute operation on target device
                result = operation(encoded_data, device)
                
                # Success - return result
                return result
                
            except Exception as e:  # pylint: disable=broad-exception-caught
                if self._is_device_limitation_error(e, device):
                    # Known device limitation - try next device
                    self.logger.warning(
                        "Device %s limitation encountered: %s. Trying fallback.", 
                        device, str(e)
                    )
                    continue
                else:
                    # Unexpected error - propagate
                    raise
        
        # All devices failed
        raise RuntimeError("All devices failed to execute operation")
    
    def _is_device_limitation_error(self, error: Exception, device: str) -> bool:
        """Check if error is a known device limitation."""
        error_str = str(error)
        
        if device == "tpu":
            return ("UNIMPLEMENTED" in error_str and "X64 element types" in error_str) or \
                   ("int64" in error_str.lower()) or \
                   ("dynamic shape" in error_str.lower())
        
        elif device == "gpu":
            return ("out of memory" in error_str.lower()) or \
                   ("cuda" in error_str.lower() and "error" in error_str.lower())
        
        else:
            return False
```

## Performance Monitoring

### Cross-Device Metrics
```python
class HybridPerformanceMonitor:
    """Monitor performance across hybrid device architecture."""
    
    def track_operation(self, operation_name: str, device: str, 
                       data_size: int, execution_time: float):
        """Track operation performance across devices."""
        
        self.metrics[operation_name][device].append({
            'data_size': data_size,
            'execution_time': execution_time,
            'throughput': data_size / execution_time,
            'timestamp': time.time()
        })
    
    def get_optimal_device_recommendation(self, operation_name: str, 
                                        data_size: int) -> str:
        """Recommend optimal device based on historical performance."""
        
        device_scores = {}
        
        for device, metrics in self.metrics[operation_name].items():
            # Find similar data sizes
            similar_metrics = [
                m for m in metrics 
                if abs(m['data_size'] - data_size) / data_size < 0.2
            ]
            
            if similar_metrics:
                avg_throughput = sum(m['throughput'] for m in similar_metrics) / len(similar_metrics)
                device_scores[device] = avg_throughput
        
        # Return device with highest throughput
        return max(device_scores.items(), key=lambda x: x[1])[0] if device_scores else "cpu"
```

## Testing Strategies

### Multi-Device Test Patterns
```python
@pytest.mark.parametrize("device", ["cpu", "gpu", "tpu"])
def test_operation_across_devices(device):
    """Test operation consistency across all available devices."""
    
    if device == "tpu" and not tpu_available():
        pytest.skip("TPU not available")
    
    # Test with device-appropriate data encoding
    encoder = MultiDeviceStateEncoder()
    test_data = create_test_data()
    encoded_data = encoder.encode_for_device(test_data, device)
    
    # Execute operation
    result = execute_operation(encoded_data, device)
    
    # Verify result consistency (accounting for precision differences)
    expected = get_expected_result(test_data)
    assert_results_equivalent(result, expected, device_tolerance=get_tolerance(device))

def assert_results_equivalent(actual, expected, device_tolerance):
    """Assert results are equivalent within device-specific tolerance."""
    
    if device_tolerance["type"] == "exact":
        assert jnp.array_equal(actual, expected)
    elif device_tolerance["type"] == "approximate":
        assert jnp.allclose(actual, expected, rtol=device_tolerance["rtol"])
    elif device_tolerance["type"] == "hash_collision_tolerant":
        # For hash operations, allow for some collisions
        unique_actual = len(jnp.unique(actual))
        unique_expected = len(jnp.unique(expected))
        collision_rate = abs(unique_actual - unique_expected) / unique_expected
        assert collision_rate < device_tolerance["max_collision_rate"]
```

## Best Practices

1. **Always profile before optimizing** - Device performance can vary significantly based on data characteristics
2. **Design for graceful degradation** - Every operation should have a CPU fallback
3. **Monitor cross-device data transfer costs** - Sometimes CPU-only is faster than hybrid due to transfer overhead
4. **Use approximate algorithms on TPU when possible** - Exact algorithms often require CPU fallback
5. **Batch operations to amortize device switching costs** - Avoid frequent device switches for small operations
6. **Cache device capability detection** - Device detection can be expensive, cache results
7. **Test with realistic data sizes** - Performance characteristics change dramatically with scale