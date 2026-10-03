"""Final-test graph statistics and parameter counts, without changing pool outputs."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class StructuralStatistics:
    """Aggregate final-test graph sizes and unweighted per-graph node retention."""
    total_input_nodes: int = 0
    total_output_nodes: int = 0
    total_input_edges: int = 0
    total_output_edges: int = 0
    total_input_nonzero_edges: int = 0
    total_output_nonzero_edges: int = 0
    num_graphs: int = 0
    _node_retention_sum: float = 0.0

    def observe(
        self,
        *,
        input_x: torch.Tensor,
        input_edge_index: torch.Tensor,
        input_batch: torch.Tensor,
        output_x: torch.Tensor,
        output_edge_index: torch.Tensor,
        output_batch: torch.Tensor,
        input_edge_weight: torch.Tensor | None = None,
        output_edge_weight: torch.Tensor | None = None,
    ) -> None:
        """Accumulate stored/nonzero edges and cluster-slot counts for one contiguous graph batch."""
        if input_x.size(0) == 0:
            raise ValueError("Cannot measure pooling structure for an empty graph batch.")

        graph_count = int(input_batch.max().item()) + 1
        input_nodes_per_graph = torch.bincount(input_batch, minlength=graph_count)
        output_nodes_per_graph = torch.bincount(output_batch, minlength=graph_count)
        if output_nodes_per_graph.numel() != graph_count:
            raise ValueError("Pooled batch contains a graph index absent from the input batch.")
        if torch.any(input_nodes_per_graph == 0):
            raise ValueError("Input batch graph indices must be contiguous.")

        self.total_input_nodes += int(input_x.size(0))
        self.total_output_nodes += int(output_x.size(0))
        self.total_input_edges += int(input_edge_index.size(1))
        self.total_output_edges += int(output_edge_index.size(1))
        self.total_input_nonzero_edges += _count_nonzero_edges(input_edge_index, input_edge_weight)
        self.total_output_nonzero_edges += _count_nonzero_edges(output_edge_index, output_edge_weight)
        self.num_graphs += graph_count
        # Average ratios per graph so large graphs do not dominate retention.
        # Dense outputs count cluster slots, so this ratio can exceed one.
        self._node_retention_sum += float(
            (
                output_nodes_per_graph.to(torch.float64)
                / input_nodes_per_graph.to(torch.float64)
            ).sum().item()
        )

    def to_mapping(self) -> dict:
        if self.num_graphs == 0:
            raise ValueError("Structural statistics observed no graphs.")
        return {
            "total_input_nodes": self.total_input_nodes,
            "total_output_nodes": self.total_output_nodes,
            "total_input_edges": self.total_input_edges,
            "total_output_edges": self.total_output_edges,
            "total_input_nonzero_edges": self.total_input_nonzero_edges,
            "total_output_nonzero_edges": self.total_output_nonzero_edges,
            "num_graphs": self.num_graphs,
            "mean_node_retention": self._node_retention_sum / self.num_graphs,
        }


def _count_nonzero_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor | None) -> int:
    # Count stored directions, self-loops, and duplicates; no thresholding or
    # coalescing, so tiny nonzero weights remain distinct from dense zero slots.
    if edge_weight is None:
        return int(edge_index.size(1))
    return int(torch.count_nonzero(edge_weight).item())


def _argument(args: tuple, kwargs: dict, name: str, position: int, default=None):
    if name in kwargs:
        return kwargs[name]
    return args[position] if position < len(args) else default


@contextmanager
def capture_structural_statistics(model):
    """Observe pooling only within this context and remove the hook even on failure."""
    statistics = StructuralStatistics()
    pool_module = getattr(model, "pool_module", None)

    if pool_module is None:
        # nopool has no module to hook; measure its identity transformation at
        # the model input instead, reporting equal input/output graph sizes.
        def observe_identity(_module, args, kwargs):
            data = _argument(args, kwargs, "data", 0)
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = data.edge_index.new_zeros(data.x.size(0))
            statistics.observe(
                input_x=data.x,
                input_edge_index=data.edge_index,
                input_batch=batch,
                output_x=data.x,
                output_edge_index=data.edge_index,
                output_batch=batch,
                input_edge_weight=getattr(data, "edge_weight", None),
                output_edge_weight=getattr(data, "edge_weight", None),
            )

        handle = model.register_forward_pre_hook(observe_identity, with_kwargs=True)
    else:
        def observe_pool(_module, args, kwargs, output):
            statistics.observe(
                input_x=_argument(args, kwargs, "x", 0),
                input_edge_index=_argument(args, kwargs, "edge_index", 1),
                input_batch=_argument(args, kwargs, "batch", 2),
                output_x=output.x,
                output_edge_index=output.edge_index,
                output_batch=output.batch,
                input_edge_weight=_argument(args, kwargs, "edge_weight", 3),
                output_edge_weight=output.edge_weight,
            )

        handle = pool_module.register_forward_hook(observe_pool, with_kwargs=True)

    try:
        yield statistics
    finally:
        handle.remove()


def count_trainable_parameters(model) -> dict:
    """Count trainable elements after lazy parameters have been materialized by training."""
    pool_module = getattr(model, "pool_module", None)
    return {
        "total": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "pooling_module": 0 if pool_module is None else sum(
            parameter.numel()
            for parameter in pool_module.parameters()
            if parameter.requires_grad
        ),
    }
