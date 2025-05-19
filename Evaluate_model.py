from graph_net import GraphResNet
from dataset import TSPDataset, Graph
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

def pass_model_a_certain_graph(num_nodes:int, density:float=1.): 
    device = 'cpu'
    model = torch.load(Path().cwd() / "baseline.pt", weights_only=False, map_location=device)        
    model.eval() 

    graph = Graph(num_nodes, density=density)
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

    # The features here have to be the same as in dataset.py, since we have to artificially create
    # a batch of one graph to pass into the model
    node_features = torch.zeros(num_nodes, 8, dtype=torch.float)
    
    node_features[:, 0] = edge_lookup.sum(dim=1)
    if node_features[:, 0].max() > 0:
        node_features[:, 1] = node_features[:, 0] / node_features[:, 0].max()
    
    node_features[:, 2] = edge_lookup.max(dim=1)[0]
    min_weights = torch.ones(num_nodes) * float('inf')
    for i in range(num_nodes):
        mask = edge_lookup[i] > 0
        if mask.any():
            min_weights[i] = edge_lookup[i, mask].min()
    min_weights[min_weights == float('inf')] = 0
    node_features[:, 3] = min_weights
    
    inv_dist = 1.0 / (edge_lookup + 1e-8)
    inv_dist[edge_lookup == 0] = 0  
    node_features[:, 4] = inv_dist.sum(dim=1)
    if node_features[:, 4].max() > 0:
        node_features[:, 5] = node_features[:, 4] / node_features[:, 4].max()
    
    node_indices = torch.arange(num_nodes, dtype=torch.float)
    node_features[:, 6] = node_indices / num_nodes  
    node_features[:, 7] = torch.sin(node_indices * (6.28 / num_nodes))  

    # Create a batch with just this single graph 
    batch_data = [(
        node_features.to(device),
        edge_index.to(device), 
        edge_attr.to(device), 
        edge_lookup.to(device)
    )]
    
    # Run inference
    with torch.no_grad():
        route, _ = model(batch_data, greedy=True)  
    
    route = route[0].cpu().numpy()
    
    print("\nInferred Route:", route)
    
    route_length = calculate_route_length(route, edge_lookup)
    print(f"Route Length: {route_length}")
    
    print(f"Number of unique nodes in route: {len(set(route[:-1]))}")  
    
    return route, route_length 

def calculate_route_length(route, edge_lookup):
    """Calculate the total length of a route using the edge lookup matrix"""
    total_length = 0
    for i in range(len(route) - 1):
        src, dst = route[i], route[i+1]
        total_length += edge_lookup[src, dst].item()
    return total_length



lens = []
for i in range(5):
    
    _, tour_len = pass_model_a_certain_graph(20)
    lens.append(tour_len)

print(f"Mean: {np.mean(lens)}")
print(f"Stdev: {np.std(lens)}")
print(f"Min: {min(lens)}")
print(f"Max: {max(lens)}")