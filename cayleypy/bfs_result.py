from dataclasses import dataclass, replace
from functools import cached_property
from typing import Optional, Union, Any
import numpy as np
import torch
from scipy.sparse import coo_array
import h5py
from cayleypy.permutation_utils import apply_permutation
from .cayley_graph_def import CayleyGraphDef


@dataclass(frozen=True)
class BfsResult:
    """Result of running breadth-first search on a Schreier coset graph.

    Can be used to obtain the graph explicitly. In this case, vertices are numbered sequentially in the order in which
    they are visited by BFS.
    """

    bfs_completed: bool  # Whether full graph was explored.
    layer_sizes: list[int]  # i-th element is number of states at distance i from start.
    layers: dict[int, torch.Tensor]  # Explicitly stored states for each layer.

    # Hashes of all vertices (if requested).
    # Order is the same as order of states in layers.
    vertices_hashes: Optional[torch.Tensor]

    # List of edges (if requested).
    # Tensor of shape (num_edges, 2) where vertices are represented by their hashes.
    edges_list_hashes: Optional[torch.Tensor]

    # Definition of the CayleyGraph on which BFS was run. Needed if we want to restore edge names.
    graph: "CayleyGraphDef"

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, BfsResult):
            return False

        if self.bfs_completed != other.bfs_completed:
            return False
        if self.layer_sizes != other.layer_sizes:
            return False
        if set(self.layers) != set(other.layers):
            return False
        for k in self.layers:
            if not torch.all(self.layers[k] == other.layers[k]):
                return False

        if not self.has_vertices_hashes():
            if other.has_vertices_hashes():
                return False
        else:
            if self.vertices_hashes.shape != other.vertices_hashes.shape:  # type: ignore
                return False
            if not torch.all(self.vertices_hashes == other.vertices_hashes):  # type: ignore
                return False

        if not self.has_edges_list_hashes():
            if other.has_edges_list_hashes():
                return False
        else:
            if self.edges_list_hashes.shape != other.edges_list_hashes.shape:  # type: ignore
                return False
            if not torch.all(self.edges_list_hashes == other.edges_list_hashes):  # type: ignore
                return False

        if self.graph != other.graph:
            return False
        return True

    def save(self, path: str):
        path = str(path)
        assert path.endswith(".h5"), "Please use '.h5' extention for BfsResult saving"

        with h5py.File(path, "w") as f:
            f["bfs_completed"] = self.bfs_completed
            f["layer_sizes"] = self.layer_sizes
            for name, layer in self.layers.items():
                f[f"layer__{name}"] = layer.detach().cpu()

            if self.has_vertices_hashes():
                f["vertices_hashes"] = self.vertices_hashes.detach().cpu()  # type: ignore
            else:
                f["vertices_hashes"] = torch.empty([])

            if self.has_edges_list_hashes():
                f["edges_list_hashes"] = self.edges_list_hashes.detach().cpu()  # type: ignore
            else:
                f["edges_list_hashes"] = torch.empty([])

            # Save graph definition based on generator type
            f["graph__generator_names"] = self.graph.generator_names
            f["graph__central_state"] = self.graph.central_state
            f["graph__generators_type"] = self.graph.generators_type.value

            if self.graph.is_permutation_group():
                f["graph__generators"] = self.graph.generators_permutations
            else:
                # Save matrix generators and their moduli
                matrices = []
                moduli = []
                for gen in self.graph.generators_matrices:
                    matrices.append(gen.matrix)
                    moduli.append(gen.modulo)
                f["graph__generator_matrices"] = np.array(matrices)
                f["graph__generator_moduli"] = np.array(moduli)

    @staticmethod
    def load(path: str):
        path = str(path)
        # pylint: disable=no-member
        with h5py.File(path, "r") as f:

            layer_sizes = f["layer_sizes"][()].tolist()

            if f["vertices_hashes"].shape == tuple():
                vertices_hashes = None
            else:
                vertices_hashes = f["vertices_hashes"][()]

            if f["edges_list_hashes"].shape == tuple():
                edges_list_hashes = None
            else:
                edges_list_hashes = f["edges_list_hashes"][()]

            # Determine generator type and create appropriate graph definition
            generator_names = [x.decode("utf-8") for x in f["graph__generator_names"][()]]
            central_state = f["graph__central_state"][()].tolist()

            # Check if this is a new format file with generator type info
            if "graph__generators_type" in f:
                from .cayley_graph_def import GeneratorType, MatrixGenerator  # pylint: disable=import-outside-toplevel

                generators_type = GeneratorType(f["graph__generators_type"][()])

                if generators_type == GeneratorType.PERMUTATION:
                    graph = CayleyGraphDef.create(
                        generators=f["graph__generators"][()].tolist(),
                        generator_names=generator_names,
                        central_state=central_state,
                    )
                else:  # MATRIX
                    matrices = f["graph__generator_matrices"][()]
                    moduli = f["graph__generator_moduli"][()]
                    generators = [MatrixGenerator.create(matrices[i], moduli[i]) for i in range(len(matrices))]
                    graph = CayleyGraphDef.for_matrix_group(
                        generators=generators,
                        generator_names=generator_names,
                        central_state=central_state,
                    )
            else:
                # Legacy format - assume permutation group
                graph = CayleyGraphDef.create(
                    generators=f["graph__generators"][()].tolist(),
                    generator_names=generator_names,
                    central_state=central_state,
                )

            loaded_result = BfsResult(
                bfs_completed=bool(f["bfs_completed"][()]),
                layer_sizes=layer_sizes,
                layers={x: f[f"layer__{str(x)}"][()] for x in range(len(layer_sizes))},
                edges_list_hashes=edges_list_hashes,
                vertices_hashes=vertices_hashes,
                graph=graph,
            )
        # pylint: enable=no-member
        return loaded_result

    def diameter(self):
        """Maximal distance from any start vertex to any other vertex."""
        return len(self.layer_sizes) - 1

    def get_layer(self, layer_id: int) -> np.ndarray:
        """Returns all states in the layer with given index."""
        if not 0 <= layer_id <= self.diameter():
            raise KeyError(f"No such layer: {layer_id}.")
        if layer_id not in self.layers:
            raise KeyError(f"Layer {layer_id} was not computed because it was too large.")
        return self.layers[layer_id].cpu().numpy()

    def last_layer(self) -> np.ndarray:
        """Returns last layer, formatted as set of strings."""
        return self.get_layer(self.diameter())

    def has_vertices_hashes(self):
        return hasattr(self, "vertices_hashes") and self.vertices_hashes is not None

    def has_edges_list_hashes(self):
        return hasattr(self, "edges_list_hashes") and self.edges_list_hashes is not None

    def to_device(self, device: Union[str, torch.device]):

        if isinstance(device, str):
            device = torch.device(device)
        # in dataclass one has to use replace() instead of regular assignment
        return replace(
            self,
            layers={k: v.to(device) for k, v in self.layers.items()},
            vertices_hashes=self.vertices_hashes.to(device) if self.has_vertices_hashes() else None,  # type: ignore
            edges_list_hashes=self.edges_list_hashes.to(device) if self.has_edges_list_hashes() else None,  # type: ignore # pylint: disable=line-too-long
        )

    @cached_property
    def num_vertices(self) -> int:
        """Number of vertices in the graph."""
        return sum(self.layer_sizes)

    @cached_property
    def hashes_to_indices_dict(self) -> dict[int, int]:
        """Dictionary used to remap vertex hashes to indexes."""
        n = self.num_vertices
        assert self.vertices_hashes is not None, "Run bfs with return_all_hashes=True."
        assert len(self.vertices_hashes) == n, "Number of vertices hashes must be the same as the number of veritces"
        ans: dict[int, int] = {}

        for i in range(n):
            ans[int(self.vertices_hashes[i])] = i
        assert len(ans) == n, "Hash collision."
        return ans

    @cached_property
    def edges_list(self) -> np.ndarray:
        """Returns list of edges, with vertices renumbered."""
        assert self.edges_list_hashes is not None, "Run bfs with return_all_edges=True."
        hashes_to_indices = self.hashes_to_indices_dict
        return np.array([[hashes_to_indices[int(h)] for h in row] for row in self.edges_list_hashes], dtype=np.int64)

    def named_undirected_edges(self) -> set[tuple[str, str]]:
        """Names for vertices (representing coset elements in readable format)."""
        vn = self.vertex_names
        return {tuple(sorted([vn[i1], vn[i2]])) for i1, i2 in self.edges_list}  # type: ignore

    def adjacency_matrix(self) -> np.ndarray:
        """Returns adjacency matrix as a dense NumPy array."""
        ans = np.zeros((self.num_vertices, self.num_vertices), dtype=np.int8)
        for i1, i2 in self.edges_list:
            ans[i1, i2] = 1
        return ans

    def adjacency_matrix_sparse(self) -> coo_array:
        """Returns adjacency matrix as a sparse SciPy array."""
        num_edges = len(self.edges_list)
        data = np.ones((num_edges,), dtype=np.int8)
        row = self.edges_list[:, 0]
        col = self.edges_list[:, 1]
        return coo_array((data, (row, col)), shape=(self.num_vertices, self.num_vertices))

    @staticmethod
    def vertex_name(state: np.ndarray) -> str:
        if len(state.shape) == 1:
            delimiter = "" if state.max() <= 9 else ","
            return delimiter.join(str(int(x)) for x in state)
        else:
            return str(state)

    @cached_property
    def vertex_names(self) -> list[str]:
        """Returns names for vertices in the graph."""
        ans = []
        for layer_id in range(len(self.layers)):
            if layer_id not in self.layers:
                raise ValueError("To get explicit graph, run bfs with max_layer_size_to_store=None.")
            for state in self.get_layer(layer_id):
                ans.append(BfsResult.vertex_name(state))
        return ans

    @cached_property
    def all_states(self) -> torch.Tensor:
        """Explicit states, ordered by index."""
        assert len(self.layers) == len(
            self.layer_sizes
        ), f"Not all layers were stored. Re-run BFS with max_layer_size_to_store={max(self.layer_sizes)}"
        return torch.vstack([self.layers[i] for i in range(len(self.layer_sizes))])

    def get_edge_name(self, i1: int, i2: int) -> str:
        """Returns name for generator used to go from vertex i1 to vertex i2."""
        state_before = self.all_states[i1].cpu().numpy()
        state_after = self.all_states[i2].cpu().numpy()
        if self.graph.is_permutation_group():
            s1 = list(state_before)
            s2 = list(state_after)
            for i in range(self.graph.n_generators):
                if apply_permutation(self.graph.generators[i], s1) == s2:
                    return self.graph.generator_names[i]
        else:
            for i, mx in enumerate(self.graph.generators_matrices):
                if np.array_equal(mx.apply(state_before), state_after):
                    return self.graph.generator_names[i]
        assert False, f"Edge ({i1},{i2}) not found."

    def to_networkx_graph(self, directed=False, with_labels=True):
        """Returns explicit graph as networkx.Graph or networkx.DiGraph."""
        # Import networkx here so we don't need to depend on this library in requirements.
        import networkx  # pylint: disable=import-outside-toplevel

        if not self.graph.generators_inverse_closed:
            assert directed, "Generators are not inverse closed, you must call to_networkx_graph(directed=True)."

        vertex_names = self.vertex_names
        ans = networkx.DiGraph() if directed else networkx.Graph()
        for name in vertex_names:
            ans.add_node(name)
        for i1, i2 in self.edges_list:
            label = self.get_edge_name(i1, i2) if with_labels else None
            ans.add_edge(vertex_names[i1], vertex_names[i2], label=label)
        return ans

    def __repr__(self):
        return f"BfsResult(diameter={self.diameter()}, layer_sizes={self.layer_sizes})"
