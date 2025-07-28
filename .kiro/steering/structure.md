# CayleyPy Project Structure

## Root Directory
- `cayleypy/`: Main package source code
- `notebooks/`: Jupyter notebooks with examples and tutorials
- `docs/`: Documentation files and build scripts
- `pyproject.toml`: Project configuration and dependencies
- `README.md`: Project overview and usage instructions
- `LICENSE`: MIT license
- `lint.sh`: Code quality checking script
- `generate_datasets.py`: Dataset generation utilities

## Main Package (`cayleypy/`)

### Core Modules
- `cayley_graph.py`: Main CayleyGraph class implementation
- `cayley_graph_def.py`: Graph definition structures and MatrixGenerator
- `graphs_lib.py`: Library of predefined graphs (PermutationGroups, MatrixGroups)
- `predictor.py`: Neural network predictor interface and pre-trained models

### Search Algorithms
- `bfs_numpy.py`: NumPy-based breadth-first search implementation
- `bfs_bitmask.py`: Optimized BFS using bitmask operations
- `beam_search_result.py`: Results container for beam search operations

### Utilities
- `permutation_utils.py`: Permutation manipulation utilities
- `string_encoder.py`: String encoding/decoding for state representation
- `hasher.py`: Hash functions for duplicate detection
- `torch_utils.py`: PyTorch utility functions
- `datasets.py`: Dataset loading and management

### Puzzles (`cayleypy/puzzles/`)
- `puzzles.py`: Main Puzzles class with common puzzle definitions
- `cube.py`: Rubik's Cube implementations (3x3x3, 4x4x4, 5x5x5)
- `globe.py`: Globe puzzle implementation
- `hungarian_rings.py`: Hungarian Rings puzzle
- `gap_puzzles.py`: GAP format puzzle loader
- `moves.py`: Hardcoded permutation moves for various puzzles
- `gap_files/`: GAP format puzzle definitions

### Models (`cayleypy/models/`)
- Neural network architectures and pre-trained model configurations
- Model library with Kaggle-hosted weights

### Data (`cayleypy/data/`)
- CSV files and other data resources

## Notebooks Structure
- `01_getting_started.ipynb`: Basic usage tutorial
- `cayleypy-demo.ipynb`: Comprehensive demonstration
- `beam-search-with-cayleypy.ipynb`: Beam search examples
- Various specialized notebooks for specific graphs and applications

## Testing Convention
- Test files follow `*_test.py` naming pattern
- Tests are co-located with source files
- Use pytest for all testing

## Documentation Structure
- `docs/CayleyPy.md`: Comprehensive project documentation
- API documentation generated with Sphinx
- Hosted at: https://cayleypy.github.io/cayleypy-docs/api.html

## Key Architectural Patterns
- **Factory pattern**: `prepare_graph()` function for graph creation
- **Strategy pattern**: Different BFS implementations (numpy, bitmask)
- **Builder pattern**: CayleyGraphDef for graph construction
- **Library pattern**: Static methods in PermutationGroups, MatrixGroups, Puzzles classes