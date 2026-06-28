"""Example custom pooling plugin for Graph Pooling Lab (GPLab).

This example demonstrates the PoolOutput contract for custom pooling plugins.
All custom pooling methods must return a PoolOutput instance from forward().

Usage:
    gplab-train --pool examples.custom_pool_plugin:build_pool --pool-ratio 0.6
"""

import torch
from torch_geometric.nn.pool import TopKPooling

from gplab.layers.pool.contracts import PoolOutput


class CustomTopKPool(torch.nn.Module):
    """
    Example custom pooling that wraps TopKPooling and returns PoolOutput.
    
    This demonstrates the required contract:
    1. Inherit from torch.nn.Module
    2. forward() accepts x, edge_index, batch
    3. Return PoolOutput with at least x, edge_index, batch
    """
    def __init__(self, in_channels: int, ratio: float = 0.5):
        super().__init__()
        self.pool = TopKPooling(in_channels, ratio=ratio)
    
    def forward(self, x, edge_index, batch):
        # Call the underlying pooling
        x_out, edge_index_out, edge_attr_out, batch_out, perm, score = self.pool(
            x=x, edge_index=edge_index, batch=batch
        )
        
        # Return PoolOutput - required contract
        return PoolOutput(
            x=x_out,
            edge_index=edge_index_out,
            batch=batch_out,
            edge_attr=edge_attr_out,
            perm=perm,
            score=score,
        )
    
    def reset_parameters(self):
        self.pool.reset_parameters()


def build_pool(
    in_channels: int,
    ratio: float = 0.5,
    avg_node_num=None,
    nonlinearity="relu",
):
    """
    Factory function for custom pooling plugin.
    
    Args:
        in_channels: Number of input feature channels
        ratio: Pooling ratio (fraction of nodes to keep)
        avg_node_num: Average nodes per graph (for dense pooling, not used here)
        nonlinearity: Activation function name (not used here)
        
    Returns:
        CustomTopKPool instance that returns PoolOutput
    """
    # `avg_node_num` and `nonlinearity` are part of Graph Pooling Lab's full plugin API.
    return CustomTopKPool(in_channels, ratio=ratio)


# Alternative: Minimal custom pooling example (for documentation)
class MinimalCustomPool(torch.nn.Module):
    """
    Minimal example showing the PoolOutput contract.
    
    Required fields in PoolOutput:
        - x: Node features [N_pooled, F]
        - edge_index: Edge indices [2, E_pooled]  
        - batch: Batch vector [N_pooled]
        
    Optional fields:
        - edge_attr: Edge features
        - perm: Selected node indices
        - score: Node selection scores
        - aux_loss: Auxiliary loss (scalar tensor)
    """
    def __init__(self, in_channels: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
        # Your learnable parameters here
        
    def forward(self, x, edge_index, batch):
        # Your pooling logic here
        # ... compute pooled_x, pooled_edge_index, pooled_batch ...
        
        # Must return PoolOutput
        return PoolOutput(
            x=x,  # Replace with actual pooled features
            edge_index=edge_index,  # Replace with actual pooled edges
            batch=batch,  # Replace with actual batch vector
            aux_loss=None,  # Optional auxiliary loss
        )

    def reset_parameters(self):
        pass
