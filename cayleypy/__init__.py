from .beam_search_result import BeamSearchResult
from .bfs_bitmask import bfs_bitmask
from .bfs_numpy import bfs_numpy
from .bfs_result import BfsResult
from .cayley_graph import CayleyGraph
from .cayley_graph_def import CayleyGraphDef, MatrixGenerator
from .datasets import load_dataset
from .graphs_lib import prepare_graph, PermutationGroups, MatrixGroups
from .predictor import Predictor
from .puzzles import Puzzles, GapPuzzles

# TPU acceleration components (optional, requires JAX)
try:
    from .tpu_backend import TPUBackend, get_tpu_backend, is_tpu_available
    from .tpu_tensor_ops import TPUTensorOpsModule, create_tpu_tensor_ops

    __all__ = [
        "BeamSearchResult",
        "bfs_bitmask",
        "bfs_numpy",
        "BfsResult",
        "CayleyGraph",
        "CayleyGraphDef",
        "MatrixGenerator",
        "load_dataset",
        "prepare_graph",
        "PermutationGroups",
        "MatrixGroups",
        "Predictor",
        "Puzzles",
        "GapPuzzles",
        "TPUBackend",
        "get_tpu_backend",
        "is_tpu_available",
        "TPUTensorOpsModule",
        "create_tpu_tensor_ops",
    ]
except ImportError:
    # JAX not available, TPU components not imported
    __all__ = [
        "BeamSearchResult",
        "bfs_bitmask",
        "bfs_numpy",
        "BfsResult",
        "CayleyGraph",
        "CayleyGraphDef",
        "MatrixGenerator",
        "load_dataset",
        "prepare_graph",
        "PermutationGroups",
        "MatrixGroups",
        "Predictor",
        "Puzzles",
        "GapPuzzles",
    ]
