import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv

class SkipConnection(nn.Module):

    """
    A residual connection layer with dimension autocast
    """

    def __init__(self, in_dims:int, out_dims:int): 
        super().__init__()

        if in_dims != out_dims: 
            self.use_projection = True
            self.projection = nn.Sequential(
                nn.Linear(in_dims, out_dims, bias=False)
            )
        else: 
            self.use_projection = False 

    def forward(self, x): 
        if self.use_projection: 
            return self.projection(x)
        else: 
            return x


class GATBlock(nn.Module): 
    """
    Simplified GAT block
    """
    def __init__(
            self, 
            in_channels=2, 
            out_channels=512, 
            dropout=0.,
            heads=16): 
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.gat = GATv2Conv(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            heads=self.heads, 
            edge_dim=1, 
            dropout=dropout,
            concat=False
        )
        
        self.residual = SkipConnection(self.in_channels, self.out_channels)

    def forward(self, batch_data): 
        x, edge_index, edge_attr, batch = batch_data
        x_orig = x

        # Simple GAT forward pass
        x, attn = self.gat(x, edge_index, edge_attr, return_attention_weights=True)
        x = F.relu(x)
        
        # Add residual connection
        res = self.residual(x_orig)
        # this really doesn't need to change 
        x = x  + res

        return x, edge_index, edge_attr, attn, batch


class ParallelHeadFusion(nn.Module): 
    """
    Simplified fusion module
    """
    def __init__(
            self, 
            in_channels, 
            out_channels_init, 
            num_heads:int=3,
            heads_per_GAT=16, 
            dropout=0.1):
        super().__init__()

        # create num_blocks GATBlocks for the head
        self.blocks = nn.ModuleList()
        # creates N parallel heads
        self.final_channels = ((2 ** num_heads)  * out_channels_init) #  8 * 64 = 512 final dimension 
        for i in range(num_heads):
                # each head has num_blocks blocks operating on the same output chanenls
            self.blocks.append(
                # append several blocks
                nn.Sequential(
                GATBlock(in_channels, out_channels_init, dropout=dropout, heads=heads_per_GAT),
                # this linear is going to project everything into 
                nn.Linear(in_features=out_channels_init, out_features=self.final_channels)
                )
            )
            out_channels_init *= 2
        self.fusion = nn.Linear(self.final_channels * num_heads, self.final_channels)

    def forward(self, batch_data):
        outputs = []
        attentions = []

        for block in self.blocks:
            
            _gat = block[0] 
            x, edge_idx, edge_attr, attn, batch_obj = _gat(batch_data)
            _rest = nn.Sequential(*list(block.children())[1:])

            x = _rest(x)
            
            outputs.append((x, edge_idx, edge_attr,batch_obj))
            attentions.append(attn)

        x_list, edge_idx_list, edge_attr_list, batch_list = zip(*outputs)

        x_concat = torch.cat(x_list, dim=-1)
        x_fused = F.relu(self.fusion(x_concat))

        edge_attr_stack = torch.stack(edge_attr_list, dim=-1)
        edge_fusion = nn.Linear(len(edge_attr_list), 1, device=x_fused.device)
        edge_attr_fused = F.relu(edge_fusion(edge_attr_stack)).squeeze(-1)

        return x_fused, edge_idx_list[0], edge_attr_fused,attentions, batch_list[0]
    

class EncodeArray(nn.Module): 
    """
    Simplified encoder
    """
    def __init__(
            self, 
            init_channels: int, 
            hidden_dim_list: list[int],
            dropout:float=0.,
            heads:int=4
            ): 
        super().__init__()

        self.init_channels = init_channels 
        self.dropout = dropout
        self.heads = heads
        
        # Create a simple sequential stack of fusion blocks
        # okay, this is a very stupid legacy code, but it's gonna stay, because reasons
        self.blocks = nn.ModuleList()
        in_dims = self.init_channels
        for dim in hidden_dim_list:
            self.blocks.append(ParallelHeadFusion(
                in_channels=in_dims, 
                out_channels_init=dim,
                heads_per_GAT=heads,
                dropout=dropout
            ))
            in_dims = dim 
        
        self.final_dim = 512 # THIS IS HARDCODED, NOT DYNAMIC, IT WILL BREAK THINGS IF YOU CHANGE THE ARCHITECTURE

    def forward(self, batch_data): 
        data_list = [
            Data(x=coords, edge_index=eidx, edge_attr=eattr) 
            for coords, eidx, eattr, _ in batch_data
        ]
        batch_obj = Batch.from_data_list(data_list)
        x, edge_idx, edge_attr, batch = batch_obj.x, batch_obj.edge_index, batch_obj.edge_attr, batch_obj.batch
        
        # this is fine, since there's going to be only one block - paralel head fusion
        # but that fusion has several heads working in tandem, already automatically fusing shit
        attn = None
        for block in self.blocks:
            x, edge_idx, edge_attr, attn, batch = block((x, edge_idx, edge_attr, batch))

        # Prepare node embeddings
        B = batch.max().item() + 1
        node_counts = torch.bincount(batch, minlength=B)
        # I could technically also make this larger? So that the model has room for revisits
        N_max = node_counts.max().item()
        
        # Create padded output and mask
        padded = torch.zeros(B, N_max, self.final_dim, device=x.device)
        mask = torch.zeros(B, N_max, dtype=torch.bool, device=x.device)
        
        # this has to stay, it prevents things from shitting themseleves due to variable graph sizes
        ptr = 0
        for i, cnt in enumerate(node_counts):
            padded[i, :cnt] = x[ptr:ptr+cnt]
            mask[i, :cnt] = True
            ptr += cnt
            
        return padded, mask, attn  # Empty list for permutations


class LSTMDecoder(nn.Module): 
    def __init__(self, embed_dim): 
        super().__init__()
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        # self attention of the LSTM cell, probably slightly too simplistic
        self.attn = nn.Linear(embed_dim, embed_dim)
        
        self.enc_attn_wt = nn.Parameter(torch.tensor(0.75)) 
        self.revisit_penalty = nn.Parameter(torch.tensor(-0.5))
        self.shortcut_weight = nn.Parameter(torch.tensor(10.0))
        self.shortcut_bonus = nn.Parameter(torch.tensor(15.0)) # a huge-ish bonus to for finding a benefitial shortcut

        self.allow_exploration = True


    def forward(self, node_emb, mask, encoder_attn = None, edge_weights=None, greedy=False):
        B, N, D = node_emb.size()
        device = node_emb.device
        
        # Initialize LSTM state
        h = torch.zeros(B, D, device=device)
        c = torch.zeros(B, D, device=device)
        
        # Start with mean of node embeddings as input
        inp = node_emb.mean(dim=1)
        
        # Track which nodes have been visited
        # Important: This tensor will not receive gradients
        visited_nodes = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        unique_visited_count = torch.zeros(B, dtype=torch.long, device=device)
        
        # Store tour indices and log probabilities
        tours = []
        log_probs = []
        
        # Variables to track first selected node and last selected node
        first_node = None
        prev_node = None

        real_nodes_count = torch.sum(mask, dim=1)
        
        for step in range(N + 1):
            if step == N:
                # Last step: return to start node to close the tour
                curr_node = first_node
                curr_log_p = torch.zeros(B, device=device)
            else:
                # Update LSTM state
                h, c = self.lstm(inp, (h, c))
                
                # Attention mechanism
                query = h.unsqueeze(1)  # [B, 1, D]
                keys = self.attn(node_emb)  # [B, N, D]
                
                # Calculate attention scores
                # transformer-like attention
                scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / (D ** 0.5)  # [B, N] 
                
                if encoder_attn is not None and prev_node is not None: 
                    # incorporate the encoder's attention into the calculation
                    # attention is returned raw by the encoders, along with edge index, so we can actually use that fucker

                    for layer_attention in encoder_attn:
                        
                        edge_idx, attn_wts = layer_attention
                        
                        if attn_wts.dim() > 1 and attn_wts.size(1) > 1:
                            attn_wts = attn_wts.mean(dim=1)


                        source_nodes = edge_idx[0]
                        tgt_nodes    = edge_idx[1]

                        for b in range(B): 
                            if prev_node is not None: 
                                prev_idx = prev_node[b].item()
                                mask_from_prev = (source_nodes == prev_idx)

                                if mask_from_prev.any(): 
                                    targets = tgt_nodes[mask_from_prev]
                                    edge_attn  = attn_wts[mask_from_prev]
                                    
                                    for t, w in zip(targets, edge_attn):
                                        if t < N: 
                                            scores[b, t] += self.enc_attn_wt * w

                # route enforcement checks 
                for b in range(B):
                    # Count unique visited nodes
                    actual_n = real_nodes_count[b].item()
                    
                    if unique_visited_count[b] < actual_n:
                        # Still have unvisited nodes - penalize revisits
                        # this is the part that ensures that all unique nodes form the graph are present
                        revisit_mask = visited_nodes[b] & mask[b]
                        scores[b].masked_fill_(revisit_mask, self.revisit_penalty)
                    else:
                        # All nodes visited, consider shortcuts based on distance
                        if edge_weights is not None and prev_node is not None:
                            prev_idx = prev_node[b].item()

                            next_idx = first_node[b].item()

                            if step < N-1 and len(tours) > step:
                                next_idx = tours[step][b].item()

                            #i.e. no shortcut cost                            
                            direct_cost = edge_weights[b, prev_idx, next_idx]

                            for potential_idx in range(N):
                                # don't consider nodes that aren't real
                                if not mask[b, potential_idx]:
                                    continue
                                # if the candidate is the same as direct index, skip
                                if potential_idx == next_idx: 
                                    continue

                                detour_cost = edge_weights[b, prev_idx, potential_idx] + edge_weights[b, potential_idx, next_idx]
                                if detour_cost < direct_cost: 
                                    savings_ratio = (direct_cost - detour_cost) / direct_cost
                                    scores[b, potential_idx] += self.shortcut_bonus * savings_ratio

                            edge_costs = edge_weights[b, prev_idx]
                            if edge_costs.max() > edge_costs.min(): 
                                normd_costs = 1. - (edge_costs - edge_costs.min()) / (edge_costs.max() - edge_costs.min())
                                scores[b] = scores[b] + self.shortcut_weight * normd_costs

                # invalid node masking
                scores = scores.masked_fill(~mask, -1e9)
                                
                # Get probabilities
                probs = F.softmax(scores, dim=1)
                
                # Sample next node
                if greedy:
                    _, curr_node = probs.max(dim=1)
                    #print(f"from greedy call: {curr_node}")
                else:
                    curr_node = torch.multinomial(probs, 1).squeeze(-1)
                
                # Get log probability
                curr_log_p = torch.log(torch.gather(probs, 1, curr_node.unsqueeze(1)).squeeze(1) + 1e-10)
                
                # Remember first selected node
                if step == 0:
                    first_node = curr_node.clone()
            
            # Add current node to tour
            tours.append(curr_node)
            log_probs.append(curr_log_p)
            
            if step < N:
                # Update state for next step
                # Create new tensor for visited nodes 
                new_visited = visited_nodes.clone()
                
                # For each batch item, mark current node as visited
                for b in range(B):
                    # If this is a new node, increment unique count
                    if not new_visited[b, curr_node[b]]:
                        unique_visited_count[b] += 1
                    
                    # Mark as visited
                    new_visited[b, curr_node[b]] = True
                
                # Update visited nodes tensor
                visited_nodes = new_visited
                
                # Set input for next step
                inp = node_emb[torch.arange(B, device=device), curr_node]
                
                # Update previous node
                prev_node = curr_node
        return torch.stack(tours, dim=1), torch.stack(log_probs, dim=1)
    
class GraphResNet(nn.Module):
    """
    Simplified graph model for TSP
    """
    def __init__(
            self,
            hidden_dims:list[int],
            init_channels:int=2,
            heads:int=4, 
            dropout:float=0., 
            greedy:bool=False
            ): 
        super().__init__()

        self.encoder = EncodeArray(init_channels, hidden_dims, dropout, heads)
        self.decoder = LSTMDecoder(self.encoder.final_dim)
        self.greedy = greedy

    def forward(self, batch_data, greedy=None): 
        # Get node embeddings and mask
        node_emb, mask, encoder_attention = self.encoder(batch_data)
        
        # Extract edge weights for distance-based decisions
        full_lookups = [sample[3] for sample in batch_data]
        B, N_max, _ = node_emb.size()
        batched_weights = torch.zeros(B, N_max, N_max, device=node_emb.device)
        
        # Process each batch item
        for i, lookup in enumerate(full_lookups):
            n = lookup.size(0)
            batched_weights[i, :n, :n] = lookup
        
        # Use greedy decoding if specified
        use_greedy = greedy if greedy is not None else self.greedy
        
        # Generate tour
        return self.decoder(node_emb, mask, edge_weights=batched_weights,encoder_attn=encoder_attention, greedy=use_greedy)