from torch import Tensor

from .classifier_base import GraphClassifierBase


class GraphClassifierSum(GraphClassifierBase):
    def _readout_before_pool(self, x: Tensor, batch: Tensor) -> Tensor:
        return self._readout_with_optional_checkpoint(x, batch)

    def _merge_graph_embeddings(self, before_pool: Tensor, after_pool: Tensor) -> Tensor:
        return before_pool + after_pool
