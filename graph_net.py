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
    A block with GATv2conv and residual connection 
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

        x, attn = self.gat(x, edge_index, edge_attr, return_attention_weights=True)
        x = F.relu(x)
        
        # Add residual connection
        res = self.residual(x_orig)
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
                # this linear is going to project everything into a common dimension
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
    processes the outputs from ParalelHEadFusion and returns them
    """
    def __init__(
            self, 
            init_channels: int, 
            hidden_dim_list: list[int],
            dropout:float=0.,
            heads:int=4,
            fixed_nodes:int=20
            ): 
        super().__init__()

        self.init_channels = init_channels 
        self.dropout = dropout
        self.heads = heads
        self.fixed_nodes = fixed_nodes
        
        # Keep ParallelHeadFusion blocks
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
        
        self.final_dim = 512

    def forward(self, batch_data): 
        data_list = [
            Data(x=coords, edge_index=eidx, edge_attr=eattr) 
            for coords, eidx, eattr, _ in batch_data
        ]
        batch_obj = Batch.from_data_list(data_list)
        x, edge_idx, edge_attr, batch = batch_obj.x, batch_obj.edge_index, batch_obj.edge_attr, batch_obj.batch
        
        attn = None
        for block in self.blocks:
            x, edge_idx, edge_attr, attn, batch = block((x, edge_idx, edge_attr, batch))

        B = batch.max().item() + 1
        
        # all graphs have exactly self.fixed_nodes nodes
        node_emb = torch.zeros(B, self.fixed_nodes, self.final_dim, device=x.device)
        
        # fixed mask (all True for fixed-size graphs), just now noticed that this doesn't really do anything? 
        # a leftover
        mask = torch.ones(B, self.fixed_nodes, dtype=torch.bool, device=x.device)
        
        # process node embeddings batch by batch
        ptr = 0
        for i in range(B):
            node_count = (batch == i).sum().item()
            
            # all graphs should have exactly self.fixed_nodes nodes
            assert node_count == self.fixed_nodes, f"Expected {self.fixed_nodes} nodes, got {node_count}"
            
            node_emb[i] = x[ptr:ptr+self.fixed_nodes]
            ptr += self.fixed_nodes
            
        return node_emb, mask, attn


class LSTMDecoder(nn.Module): 
    def __init__(self, embed_dim, fixed_nodes=20): 
        super().__init__()
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.attn = nn.Linear(embed_dim, embed_dim)
        
        self.fixed_nodes = fixed_nodes
        
        # attention weights and learnable parameters
        self.enc_attn_wt = nn.Parameter(torch.tensor(0.5)) 
        self.revisit_penalty = nn.Parameter(torch.tensor(-0.05))  
        self.shortcut_weight = nn.Parameter(torch.tensor(10.0))
        self.shortcut_bonus = nn.Parameter(torch.tensor(30.0))  
        
        # allow longer tours for shortcut exploration
        self.exploration_ratio = 1.5
        
        # "done" prediction for deciding when to end the tour
        self.done_predictor = nn.Linear(embed_dim, 1)

    def forward(self, node_emb, mask, encoder_attn=None, edge_weights=None, greedy=False):
        B, N, D = node_emb.size()
        device = node_emb.device
        
        # sanity check yanked in after moving away from varialbe sized graphs
        assert N == self.fixed_nodes, f"Expected {self.fixed_nodes} nodes, got {N}"
        
        # calculate maximum steps with exploration buffer
        max_steps = int(N * self.exploration_ratio)
        
        # initialize LSTM state
        h = torch.zeros(B, D, device=device)
        c = torch.zeros(B, D, device=device)
        
        # mean of node embeddings as input
        inp = node_emb.mean(dim=1)
        
        # track which nodes have been visited
        visited_nodes = torch.zeros(B, N, dtype=torch.bool, device=device)
        unique_visited_count = torch.zeros(B, dtype=torch.long, device=device)
        
        # track when each batch item's tour is complete
        tour_complete = torch.zeros(B, dtype=torch.bool, device=device)
        
        # store tour indices and log probabilities
        tours = []
        log_probs = []
        
        # variables to track first selected node and last selected node
        first_node = None
        prev_node = None
        
        # dictionary to track planned shortcuts
        planned_shortcuts = {}
        
        for step in range(max_steps + 1):
            # check if all tours are complete
            if tour_complete.all():
                break
                
            # force return to start
            if step == max_steps:
                curr_node = torch.zeros(B, dtype=torch.long, device=device)
                for b in range(B):
                    if not tour_complete[b]:
                        curr_node[b] = first_node[b]
                        tour_complete[b] = True
                curr_log_p = torch.zeros(B, device=device)
            else:
                # update LSTM state
                h, c = self.lstm(inp, (h, c))
                
                # should we complete the tour?
                for b in range(B):
                    if not tour_complete[b]:
                        if unique_visited_count[b] >= N and step >= N:
                            done_logit = self.done_predictor(h[b:b+1])
                            done_prob = torch.sigmoid(done_logit)
                            
                            if greedy:
                                should_complete = done_prob > 0.5
                            else:
                                should_complete = torch.rand(1, device=device) < done_prob
                                
                            if should_complete:
                                tour_complete[b] = True
                
                # Attention mechanism
                query = h.unsqueeze(1)  # [B, 1, D]
                keys = self.attn(node_emb)  # [B, N, D]
                
                # Calculate attention scores
                scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / (D ** 0.5)
                
                # Incorporate encoder attention if available
                if encoder_attn is not None and prev_node is not None: 
                    for layer_attention in encoder_attn:
                        edge_idx, attn_wts = layer_attention
                        
                        if attn_wts.dim() > 1 and attn_wts.size(1) > 1:
                            attn_wts = attn_wts.mean(dim=1)

                        source_nodes = edge_idx[0]
                        tgt_nodes = edge_idx[1]

                        for b in range(B): 
                            if prev_node is not None and not tour_complete[b]: 
                                prev_idx = prev_node[b].item()
                                mask_from_prev = (source_nodes == prev_idx)

                                if mask_from_prev.any(): 
                                    targets = tgt_nodes[mask_from_prev]
                                    edge_attn = attn_wts[mask_from_prev]
                                    
                                    for t, w in zip(targets, edge_attn):
                                        if t < N: 
                                            scores[b, t] += self.enc_attn_wt * w
                
                # Check for planned shortcuts for this step
                has_shortcut = torch.zeros(B, dtype=torch.bool, device=device)
                forced_nodes = torch.zeros(B, dtype=torch.long, device=device)
                
                for b in range(B):
                    # if this batch has a planned shortcut for this step
                    if b in planned_shortcuts and planned_shortcuts[b][0] == step:
                        # get the next node in the shortcut sequence
                        next_shortcut_node = planned_shortcuts[b][1][0]
                        forced_nodes[b] = next_shortcut_node
                        has_shortcut[b] = True
                        
                        # update the shortcut plan
                        remaining_nodes = planned_shortcuts[b][1][1:]
                        if len(remaining_nodes) > 0:
                            planned_shortcuts[b] = (step + 1, remaining_nodes)
                        else:
                            del planned_shortcuts[b]
                
                # dynamic shortcut detection
                for b in range(B):
                    if tour_complete[b] or has_shortcut[b]:
                        continue
                    
                    # ensure all nodes are visited
                    if unique_visited_count[b] < N:
                        # revisit penalty to encourage covering all nodes first
                        revisit_mask = visited_nodes[b]
                        scores[b].masked_fill_(revisit_mask, self.revisit_penalty)
                    else:
                        #all nodes visited - actively look for beneficial shortcuts
                        if edge_weights is not None and prev_node is not None:
                            prev_idx = prev_node[b].item()
                            
                            next_idx = first_node[b].item()  
                            
                            # try to look ahead in the existing tour
                            if step < len(tours) and step < N-1:
                                next_idx = tours[step][b].item()
                            
                            # cost of direct path (similar to the shortcut optimization in the simulated annealing
                            # part of the rust repo)
                            direct_cost = edge_weights[b, prev_idx, next_idx]
                            
                            # best possible shortcut
                            best_savings = 0.0
                            best_shortcut = None
                            
                            # check all potential intermediate nodes
                            for mid_idx in range(N):
                                if mid_idx == next_idx:
                                    continue
                                    
                                via_mid = edge_weights[b, prev_idx, mid_idx] + \
                                          edge_weights[b, mid_idx, next_idx]
                                
                                if via_mid < direct_cost:
                                    savings = direct_cost - via_mid
                                    if savings > best_savings:
                                        best_savings = savings
                                        best_shortcut = [mid_idx, next_idx]
                            
                            #good shortcut? plan it
                            if best_shortcut is not None and best_savings > 0:
                                planned_shortcuts[b] = (step, best_shortcut)
                                forced_nodes[b] = best_shortcut[0]
                                has_shortcut[b] = True
                            else:
                                # just add bonuses for individual nodes proportional to savings
                                for potential_idx in range(N):
                                    if potential_idx == next_idx:
                                        continue
                                    
                                    detour_cost = edge_weights[b, prev_idx, potential_idx] + \
                                                 edge_weights[b, potential_idx, next_idx]
                                    
                                    # if shortcut saves distance, add a bonus
                                    if detour_cost < direct_cost:
                                        savings_ratio = (direct_cost - detour_cost) / direct_cost
                                        scores[b, potential_idx] += self.shortcut_bonus * savings_ratio
                            
                            # Also apply regular edge cost normalization
                            edge_costs = edge_weights[b, prev_idx]
                            if edge_costs.max() > edge_costs.min():
                                normalized_costs = 1.0 - (edge_costs - edge_costs.min()) / (edge_costs.max() - edge_costs.min())
                                scores[b] = scores[b] + self.shortcut_weight * normalized_costs
                
                # Apply forced nodes for shortcuts
                for b in range(B):
                    if has_shortcut[b]:
                        # force selection of shortcut node by setting a very high score
                        scores[b] = torch.full_like(scores[b], -1e9)
                        scores[b, forced_nodes[b]] = 100.0
                
                # force return to start if tour is completed
                for b in range(B):
                    if tour_complete[b]:
                        scores[b] = torch.full_like(scores[b], -1e9)
                        scores[b, first_node[b]] = 100.0
                
                # get probabilities
                probs = F.softmax(scores, dim=1)
                
                # Sample next node or use forced node
                curr_node = torch.zeros(B, dtype=torch.long, device=device)
                for b in range(B):
                    if has_shortcut[b] or tour_complete[b]:
                        if tour_complete[b]:
                            curr_node[b] = first_node[b]
                        else:
                            curr_node[b] = forced_nodes[b]
                    elif greedy:
                        # greedy selection - used in inference adn updataes
                        _, curr_node[b] = probs[b].max(dim=0)
                    else:
                        # sample from distribution - used in trainign
                        curr_node[b] = torch.multinomial(probs[b], 1).item()
                
                # log probability
                curr_log_p = torch.log(torch.gather(probs, 1, curr_node.unsqueeze(1)).squeeze(1) + 1e-10)
                
                # Remember first selected node
                if step == 0:
                    first_node = curr_node.clone()
            
            # Add current node to tour
            tours.append(curr_node.clone()) 
            log_probs.append(curr_log_p)
            
            if step < max_steps:
                # Update state for next step
                
                # Create new tensor for visited nodes
                new_visited = visited_nodes.clone()
                
                # For each batch item, mark current node as visited
                for b in range(B):
                    if not tour_complete[b]:
                        # If this is a new node, increment unique count
                        if not new_visited[b, curr_node[b]]:
                            unique_visited_count[b] += 1
                        
                        new_visited[b, curr_node[b]] = True
                
                # Update visited nodes tensor
                visited_nodes = new_visited
                
                # Set input for next step
                inp = node_emb[torch.arange(B, device=device), curr_node]
                
                # Update previous node
                prev_node = curr_node.clone()
        
        # stack and return results
        return torch.stack(tours, dim=1), torch.stack(log_probs, dim=1)
    
class GraphResNet(nn.Module):
    """
    Simplified graph model for TSP
    the name is a misnomer from earlier parts of writing this thing
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
        node_emb, _, encoder_attention = self.encoder(batch_data)
        
        # extract edge weights for distance-based decisions
        full_lookups = [sample[3] for sample in batch_data]
        B, N_max, _ = node_emb.size()
        batched_weights = torch.zeros(B, N_max, N_max, device=node_emb.device)
        
        # process each batch item
        for i, lookup in enumerate(full_lookups):
            n = lookup.size(0)
            batched_weights[i, :n, :n] = lookup
        
        # use greedy decoding if specified
        use_greedy = greedy if greedy is not None else self.greedy
        
        # yeet out a tour
        return self.decoder(node_emb, _, edge_weights=batched_weights,encoder_attn=encoder_attention, greedy=use_greedy)