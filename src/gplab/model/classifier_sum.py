from torch import Tensor

from .classifier_base import GraphClassifierBase


class GraphClassifierSum(GraphClassifierBase):
    def _readout_before_pool(self, x: Tensor, batch: Tensor) -> Tensor:
        return self.execution.tensor_batch(self.global_pool, x, batch)

    def _merge_graph_embeddings(self, before_pool: Tensor, after_pool: Tensor) -> Tensor:
        return before_pool + after_pool
