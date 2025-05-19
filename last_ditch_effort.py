import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        output = self.out_linear(attn_output)
        
        return output, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention block
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, ff_dim, n_layers, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Process through encoder layers
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
            
        return x, attentions


class DecoderStep(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.project_out = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Parameters to guide shortcuts and revisits
        self.shortcut_bonus = nn.Parameter(torch.tensor(5.0))
        self.revisit_penalty = nn.Parameter(torch.tensor(-0.5))

    def forward(self, decoder_state, node_embeddings, mask=None, edge_weights=None, visited=None,
                prev_node=None, step=None, all_visited=None):
        """
        Args:
            decoder_state: Current LSTM/context state [batch_size, embed_dim]
            node_embeddings: Encoded node representations [batch_size, n_nodes, embed_dim]
            mask: Valid nodes mask [batch_size, n_nodes]
            edge_weights: Distance matrix [batch_size, n_nodes, n_nodes]
            visited: Tensor tracking visited nodes [batch_size, n_nodes]
            prev_node: Previous node indices [batch_size]
            step: Current decoding step
            all_visited: Boolean tensor indicating if all nodes visited [batch_size]
        """
        batch_size, n_nodes = node_embeddings.size(0), node_embeddings.size(1)
        
        # Add batch dimension to decoder state
        query = decoder_state.unsqueeze(1)
        
        # Calculate attention scores
        attn_output, attn_weights = self.attention(query, node_embeddings, node_embeddings, mask)
        
        # Squeeze output and apply projection
        attn_output = self.norm(self.dropout(attn_output.squeeze(1)))
        logits = self.project_out(attn_output)
        
        # Calculate pointing scores (compatibility with each node)
        scores = torch.bmm(logits.unsqueeze(1), node_embeddings.transpose(1, 2)).squeeze(1)
        
        # Apply revisit penalties and shortcut bonuses
        for b in range(batch_size):
            if visited is not None:
                # Phase 1: Before visiting all nodes, penalize revisits
                if not all_visited[b]:
                    # Apply revisit penalty
                    scores[b].masked_fill_(visited[b], self.revisit_penalty)
                # Phase 2: After visiting all nodes, look for shortcuts
                elif edge_weights is not None and prev_node is not None:
                    p_idx = prev_node[b].item()
                    
                    # Try to benefit from shortcuts
                    for mid_idx in range(n_nodes):
                        if edge_weights is not None:
                            # We need to figure out the destination
                            # Assume next destination is the first node to complete the tour
                            next_idx = 0  # First node by default
                            
                            # Calculate direct path cost
                            direct = edge_weights[b, p_idx, next_idx]
                            
                            # Calculate potential shortcut
                            via_mid = edge_weights[b, p_idx, mid_idx] + edge_weights[b, mid_idx, next_idx]
                            
                            # If shortcut is beneficial, add bonus proportional to savings
                            if via_mid < direct:
                                savings_ratio = (direct - via_mid) / direct
                                scores[b, mid_idx] += self.shortcut_bonus * savings_ratio
        
        return scores


class TSPTransformer(nn.Module):
    def __init__(self, input_dim=8, embed_dim=128, n_heads=8, ff_dim=512, n_layers=3, dropout=0.1):
        super().__init__()
        
        # Encoder
        self.encoder = Encoder(input_dim, embed_dim, n_heads, ff_dim, n_layers, dropout)
        
        # LSTM for maintaining state during decoding
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        
        # Decoder step module
        self.decoder_step = DecoderStep(embed_dim, n_heads, dropout)
        
        # Final projection for done prediction
        self.done_predictor = nn.Linear(embed_dim, 1)
        
        # Parameters
        self.embed_dim = embed_dim
        self.exploration_ratio = 1.5
    
    def forward(self, batch_data, greedy=False):
        """
        Compatible with dataset and REINFORCE implementation
        """
        # Extract data from batch
        node_features = torch.stack([data[0] for data in batch_data])
        edge_weights = torch.stack([data[3] for data in batch_data])
        
        batch_size, n_nodes, _ = node_features.size()
        device = node_features.device
        
        # Encode nodes
        node_embeddings, _ = self.encoder(node_features)
        
        # Compute maximum steps with exploration buffer
        max_steps = int(n_nodes * self.exploration_ratio)
        
        # Initialize decoder state
        h = torch.zeros(batch_size, self.embed_dim, device=device)
        c = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # Start with mean of node embeddings as context
        context = node_embeddings.mean(dim=1)
        
        # Track visited nodes
        visited = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
        unique_visited_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Track tour completion
        tour_complete = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Store tour and log probabilities
        tours = []
        log_probs = []
        
        # First node and previous node tracking
        first_node = None
        prev_node = None
        
        # Track if all nodes have been visited
        all_visited = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Main decoding loop
        for step in range(max_steps + 1):
            # Check if all tours are complete
            if tour_complete.all():
                break
                
            # Last step: return to first node
            if step == max_steps:
                curr_node = torch.zeros(batch_size, dtype=torch.long, device=device)
                for b in range(batch_size):
                    if not tour_complete[b]:
                        curr_node[b] = first_node[b]
                        tour_complete[b] = True
                curr_log_p = torch.zeros(batch_size, device=device)
            else:
                # Update LSTM state
                h, c = self.lstm(context, (h, c))
                
                # Check if we've visited all nodes and should consider ending
                for b in range(batch_size):
                    if not tour_complete[b] and unique_visited_count[b] >= n_nodes and step >= n_nodes:
                        done_logit = self.done_predictor(h[b:b+1])
                        done_prob = torch.sigmoid(done_logit)
                        
                        if (greedy and done_prob > 0.5) or (not greedy and torch.rand(1, device=device) < done_prob):
                            tour_complete[b] = True
                
                # Get scores from decoder step
                scores = self.decoder_step(
                    h, node_embeddings, 
                    edge_weights=edge_weights,
                    visited=visited,
                    prev_node=prev_node,
                    step=step,
                    all_visited=all_visited
                )
                
                # Mask out invalid nodes
                # No invalid nodes in this fixed-size implementation
                
                # Force completed tours to return to start
                for b in range(batch_size):
                    if tour_complete[b]:
                        scores[b] = torch.full_like(scores[b], -1e9)
                        scores[b, first_node[b]] = 100.0
                
                # Convert scores to probabilities
                probs = F.softmax(scores, dim=1)
                
                # Select next node (greedy or sampling)
                if greedy:
                    _, curr_node = probs.max(dim=1)
                else:
                    curr_node = torch.multinomial(probs, 1).squeeze(-1)
                
                # Calculate log probabilities
                curr_log_p = torch.log(torch.gather(probs, 1, curr_node.unsqueeze(1)).squeeze(1) + 1e-10)
                
                # Remember first node
                if step == 0:
                    first_node = curr_node.clone()
            
            # Add to tour
            tours.append(curr_node.clone())
            log_probs.append(curr_log_p)
            
            if step < max_steps:
                # Update visited nodes
                new_visited = visited.clone()
                
                for b in range(batch_size):
                    if not tour_complete[b]:
                        # If this is a new node, increment unique count
                        if not new_visited[b, curr_node[b]]:
                            unique_visited_count[b] += 1
                        
                        # Mark as visited
                        new_visited[b, curr_node[b]] = True
                        
                        # Check if all nodes visited
                        if unique_visited_count[b] >= n_nodes and not all_visited[b]:
                            all_visited[b] = True
                
                # Update visited tracking
                visited = new_visited
                
                # Update context with current node embedding
                context = node_embeddings[torch.arange(batch_size, device=device), curr_node]
                
                # Update previous node
                prev_node = curr_node.clone()
        
        # Stack results
        tour_tensor = torch.stack(tours, dim=1)
        log_prob_tensor = torch.stack(log_probs, dim=1)
        
        return tour_tensor, log_prob_tensor