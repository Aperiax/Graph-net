"""
module handling creating batches for dataloaders and the graph instances
for model to learn
"""


from graph_maker import Graph as grph # a sloppily compiled rust project converted into a python library
import torch 
from torch.utils.data import Dataset
import contextlib
import io



class Graph: 
    """
    Rust-based library python wrapper, since pyo3 don't play fair with pylacne
    """

    def __init__(self, num_vertices:int, density:float = 1.):
        """
        Graph constructor, 
        :param num_vertices: The wanted size of the graph
        :param density: Desired graph density. The graph.rs is capable of generating an
                        arbitrarily sparse graph, although for this project we'll just be
                        sticking with density=1., since due to time constraints the code is
                        optimized only for fully connected undirected graphs
        :out: -> A graph rust struct as a python class ->
        the full definition of the struct:
        pub struct Graph
        {
            pub vertices: Vec<Vertex>,
            pub name_to_id: HashMap<String, usize>,
            pub adjacency_matrix: Array2<usize>,
            #[pyo3(get)]
            pub size: usize,
        }
        """
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with contextlib.redirect_stdout(io.StringIO()):
            self._graph = grph(num_vertices, density)

    def get_adjacency_matrix(self): 
        """
        Retrieves the adjacency matrix as a numpy.ndarray object for work related to indexing 
        tours and printing and then subsequently returns it's tensor
        """
        return self._graph.adjacency_numpy()

    def edge_list(self): 
        """
        Retrieves the graph's edge list along with edge weights in a format of 
        list[tuple[int, int, int]] -> tuple(source, target, weight)
        """
        return self._graph.edge_list()

class TSPDataset(Dataset):
    def __init__(self, num_samples, min_nodes=10, max_nodes=100):
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        """
        Handles what is the structure of each batch
        """

        # using the randint so i can maintain the model's seed
        # handles batching the data, __getitem__ dictates how torch feeds batches into the neural net

        num_nodes = torch.randint(self.min_nodes, self.max_nodes+1, size=())
        graph = Graph(num_nodes, density=1.0)
        edge_list = graph.edge_list()  # list of (src, tgt, weight)

        # so it creates a random tensor of shape [num_nodes, 2]
        coords = torch.rand(num_nodes, 2)

        edge_index = []
        edge_attr = []
        edge_lookup = torch.zeros(num_nodes, num_nodes, dtype=torch.float)

        
        for src, tgt, weight in edge_list:
            edge_index.append([src, tgt])
            edge_attr.append([weight])

            # symmetric update. basically making adjacency matrices into tensors for fast lookup
            edge_lookup[src, tgt] = weight
            edge_lookup[tgt, src] = weight
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        node_features = torch.zeros(num_nodes, 8, dtype=torch.float)
        
        # 1-2: Weighted degree (sum of edge weights)
        node_features[:, 0] = edge_lookup.sum(dim=1)
        # Normalize
        if node_features[:, 0].max() > 0:
            node_features[:, 1] = node_features[:, 0] / node_features[:, 0].max()
        
        # 3-4: Max and min edge weights
        node_features[:, 2] = edge_lookup.max(dim=1)[0]
        # Min edge weight (non-zero)
        min_weights = torch.ones(num_nodes) * float('inf')
        for i in range(num_nodes):
            mask = edge_lookup[i] > 0
            if mask.any():
                min_weights[i] = edge_lookup[i, mask].min()
        min_weights[min_weights == float('inf')] = 0
        node_features[:, 3] = min_weights
        
        # 5-6: Closeness centrality approximation (inverse distances)
        inv_dist = 1.0 / (edge_lookup + 1e-8)
        inv_dist[edge_lookup == 0] = 0  # Zero for disconnected
        node_features[:, 4] = inv_dist.sum(dim=1)
        # Normalize
        if node_features[:, 4].max() > 0:
            node_features[:, 5] = node_features[:, 4] / node_features[:, 4].max()
        
        # 7-8: Node position encoding
        node_indices = torch.arange(num_nodes, dtype=torch.float)
        node_features[:, 6] = node_indices / num_nodes  # Normalized position
        node_features[:, 7] = torch.sin(node_indices * (6.28 / num_nodes))  # Circular position
        
        return node_features, edge_index, edge_attr, edge_lookup
        