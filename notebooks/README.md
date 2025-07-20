# CayleyPy Notebooks

This directory contains Jupyter notebooks for exploring and demonstrating CayleyPy functionality.

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

- **Getting Started**: Basic CayleyPy usage and examples
- **Puzzle Analysis**: Analyzing various combinatorial puzzles
- **Performance Benchmarks**: Comparing different algorithms and optimizations
- **Visualization**: Creating graphs and plots of Cayley graph properties
- **Research Examples**: Advanced use cases and research applications

## Dependencies

The notebooks use additional visualization and analysis libraries:
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `jupyter` - Jupyter notebook interface
- `jupyterlab` - Modern Jupyter interface
- `ipykernel` - Python kernel for Jupyter

## Contributing

When adding new notebooks:
1. Clear all outputs before committing
2. Include markdown documentation explaining the notebook's purpose
3. Add the notebook to this README
4. Test that the notebook runs from a clean environment