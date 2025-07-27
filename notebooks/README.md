# CayleyPy Notebooks

This directory contains Jupyter notebooks for exploring and demonstrating CayleyPy functionality. These notebooks showcase various applications of the library for analyzing Cayley graphs, solving combinatorial puzzles, and benchmarking performance.

## Setup

To use these notebooks, install the development dependencies:

```bash
pip install -e .[dev,torch]
```

Then start Jupyter:

```bash
jupyter lab
```

## Notebooks

### Getting Started
- **01_getting_started.ipynb** - Basic introduction to CayleyPy for analyzing Cayley graphs and solving combinatorial puzzles

### Algorithms and Search
- **beam-search-with-cayleypy.ipynb** - Demonstrates how to use beam search to find paths from random permutations to identity permutation in LRX Cayley graph
- **lrx-solution.ipynb** - Solutions for LRX permutation problems
- **solve-lrx-binary-with-cayleypy.ipynb** - Using neural networks with CayleyPy to solve binary LRX problems
- **lrx-binary-with-cayleypy-bfs-only.ipynb** - BFS-only approach to solving binary LRX problems

### Analysis and Research
- **growth-function-for-lx-cayley-graph.ipynb** - Computes growth functions for LX Cayley Graph (OEIS A039745)
- **computing-spectra-of-cayley-graphs-using-cayleypy.ipynb** - Demonstrates computing spectral properties of Cayley graphs
- **library-of-puzzles-in-gap-format-in-cayleypy.ipynb** - Explores CayleyPy's support for puzzles defined in GAP format
- **cayleypy-demo.ipynb** - General demonstration of CayleyPy capabilities

### Performance Benchmarks
- **benchmark-bfs-in-cayleypy-on-gpu-p100.ipynb** - Benchmarks BFS performance on P100 GPU
- **benchmark-versions-of-bfs-in-cayleypy.ipynb** - Compares different BFS implementations in CayleyPy

## Performance Notes

When using GPU acceleration:
- For small graphs (N ≤ 8), CPU is typically faster than GPU due to overhead
- MPS (Apple Silicon) has incomplete PyTorch operations that cause significant slowdowns
- For large graphs (N ≥ 10), GPU acceleration can provide significant benefits
- The `torch.unique()` operation used in BFS is not optimized for MPS, causing performance issues

## Dependencies

The notebooks use additional visualization and analysis libraries:
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `jupyter` - Jupyter notebook interface
- `jupyterlab` - Modern Jupyter interface
- `ipykernel` - Python kernel for Jupyter
- `torch` - PyTorch for neural network models and GPU acceleration

## Contributing

When adding new notebooks:
1. Clear all outputs before committing
2. Include markdown documentation explaining the notebook's purpose
3. Add the notebook to this README
4. Test that the notebook runs from a clean environment
5. Use CPU (`device="cpu"`) for small graphs (N ≤ 8) to avoid GPU overhead
6. Include performance considerations when using GPU acceleration