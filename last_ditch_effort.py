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
    def __init__(self, in_ch, out_ch, heads, dropout): 
        super().__init__()

        self.gat = GATv2Conv(
            in_channels=in_ch,
            out_channels=out_ch,
            heads=16,
            dropout=dropout,
            concat=False, 
            return_attention_weights=True
        )

        self.residual = SkipConnection(in_ch, out_ch)


    def forward(self, batch_data): 
        x, edge_index, edge_attr, batch = batch_data
        orig_x = x 


        gat_out, attn_tuple = self.gat(x, edge_index, edge_attr, batch)
        _, attn_weights = attn_tuple

        x = F.relu(gat_out)
        res = self.residual(orig_x)

        x = x + res 

        return x, edge_index, edge_attr, batch, attn_weights


class ParallelHeadFusion(nn.Module): 
    """
    Simplified fusion module
    """
    def __init__(self, in_channels, out_channels, num_blocks=2, heads=4, dropout=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            GATBlock(in_channels, out_channels, dropout=dropout, heads=heads)
            for _ in range(num_blocks)
        ])
        
        self.fusion = nn.Linear(out_channels * num_blocks, out_channels)

    def forward(self, batch_data):
        outputs = [block(batch_data) for block in self.blocks]
        x_list, edge_idx_list, edge_attr_list, batch_list, attn_wt_list = zip(*outputs)
        
        # Simple concatenation and linear fusion
        x_concat = torch.cat(x_list, dim=-1)
        x_fused = F.relu(self.fusion(x_concat))
        
        combined_attn = torch.mean(torch.stack(attn_wt_list), dim=0)

        # Just use the first block's edges
        return x_fused, edge_idx_list[0], edge_attr_list[0], batch_list[0], combined_attn


class EncodeArray(nn.Module): 

    def __init__(self, init_ch, hidd_dim_list, dropout=0., heads=4): 
        super().__init__()

        self.init_ch = init_ch 
        self.dropout = dropout
        self.heads   = heads

        self.init_embedding = nn.Sequential(
            nn.Linear(init_ch, hidd_dim_list[0]), 
            nn.LayerNorm(hidd_dim_list[0]),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        in_dims = hidd_dim_list[0]
        for dim in hidd_dim_list[1:]:
            self.blocks.append(ParallelHeadFusion(
                in_channels=in_dims, 
                out_channels=dim,
                num_blocks=2,
                heads=heads,
                dropout=dropout
            ))
            in_dims = dim
        
        self.final_dim = hidd_dim_list[-1]

        self.attn_proj = nn.Sequential(
            nn.Linear(heads, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_data): 
        pass