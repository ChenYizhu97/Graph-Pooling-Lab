import torch
from tgp.src import PoolingOutput
from torch import Tensor
from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from gplab.data.sparse import to_sparse_batch

from ..functional import dense_connect


class DensePoolAdapter(torch.nn.Module):
    """Adapt dense assignments to fixed cluster slots, preserving adjacency and auxiliary losses."""
    def __init__(self, assignment_layer: torch.nn.Module, pool_method: str) -> None:
        super().__init__()
        self.assignment_layer = assignment_layer
        self.pool_method = pool_method
        self.reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> PoolingOutput:
        dense_x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(adj, batch, edge_attr=edge_weight)
        assignment = (
            self.assignment_layer(dense_x, adj, mask)
            if self.pool_method == "diffpool"
            else self.assignment_layer(dense_x)
        )

        if self.pool_method == "mincutpool":
            pooled_x, pooled_adj, mincut_loss, ortho_loss = dense_mincut_pool(
                dense_x,
                adj,
                assignment,
                mask,
            )
            loss = {"mincut": 0.5 * mincut_loss, "orthogonality": ortho_loss}
        elif self.pool_method == "diffpool":
            pooled_x, pooled_adj, link_loss, ent_loss = dense_diff_pool(
                dense_x,
                adj,
                assignment,
                mask,
            )
            loss = {"link": 0.1 * link_loss, "entropy": 0.1 * ent_loss}
        elif self.pool_method == "densepool":
            pooled_x, pooled_adj = dense_connect(dense_x, adj, assignment, mask)
            loss = None
        else:
            raise ValueError(f"Unsupported dense pooling method '{self.pool_method}'.")

        sparse_x, sparse_edge_index, sparse_batch, sparse_edge_weight = to_sparse_batch(
            pooled_x,
            pooled_adj,
        )

        return PoolingOutput(
            x=sparse_x,
            edge_index=sparse_edge_index,
            batch=sparse_batch,
            edge_weight=sparse_edge_weight,
            loss=loss,
        )

    def reset_parameters(self) -> None:
        self.assignment_layer.reset_parameters()
