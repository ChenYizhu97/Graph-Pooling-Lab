from typing import Optional

from torch import Tensor

from .classifier_base import GraphClassifierBase


class GraphClassifierPlain(GraphClassifierBase):
    def _merge_graph_embeddings(self, before_pool: Optional[Tensor], after_pool: Tensor) -> Tensor:
        return after_pool
