from graph_net import GraphResNet
from dataset import TSPDataset, Graph
import torch
from torch.utils.data import DataLoader
from pathlib import Path

def pass_model_a_certain_graph(num_nodes:int, density:float=1.): 
    device = 'cpu'
    model = torch.load(Path().cwd() / "Model_trained_40.pt", weights_only=False, map_location=device)        
    model.eval() 

    # Create the graph
    graph = Graph(num_nodes, density=density)
    print("Graph Adjacency Matrix:")
    print(graph.get_adjacency_matrix())
    edge_list = graph.edge_list()  # list of (src, tgt, weight)

    # Build edge information
    edge_index = []
    edge_attr = []
    edge_lookup = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    
    for src, tgt, weight in edge_list:
        edge_index.append([src, tgt])
        edge_attr.append([weight])
        edge_lookup[src, tgt] = weight
        edge_lookup[tgt, src] = weight
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create node features (exactly as in TSPDataset)
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

    # Create a batch with just this single graph - use node_features instead of coords
    batch_data = [(
        node_features.to(device),
        edge_index.to(device), 
        edge_attr.to(device), 
        edge_lookup.to(device)
    )]
    
    # Run inference
    with torch.no_grad():
        route, _ = model(batch_data, greedy=True)  # Use greedy for evaluation
    
    # Extract the single route from the batch
    route = route[0].cpu().numpy()
    
    # Print the adjacency matrix and route
    print("\nInferred Route:", route)
    
    # Calculate route length
    route_length = calculate_route_length(route, edge_lookup)
    print(f"Route Length: {route_length}")
    
    # Calculate optimal route length (if known)
    print(f"Number of unique nodes in route: {len(set(route[:-1]))}")  # Exclude last node (return to start)
    
    return route, graph, edge_lookup, node_features

def calculate_route_length(route, edge_lookup):
    """Calculate the total length of a route using the edge lookup matrix"""
    total_length = 0
    for i in range(len(route) - 1):
        src, dst = route[i], route[i+1]
        total_length += edge_lookup[src, dst].item()
    return total_length

pass_model_a_certain_graph(15)