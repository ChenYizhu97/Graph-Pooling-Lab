from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from gplab.layers.pool.contracts import PoolOutput


class ForwardExecutor:
    def __init__(self, activation_checkpoint: bool) -> None:
        self.activation_checkpoint = activation_checkpoint

    def _enabled(self) -> bool:
        return self.activation_checkpoint and torch.is_grad_enabled()

    def tensor(self, function: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
        if not self._enabled():
            return function(x)
        return checkpoint(function, x, use_reentrant=False)

    def tensor_batch(
        self,
        function: Callable[..., Tensor],
        x: Tensor,
        batch: Tensor,
    ) -> Tensor:
        if not self._enabled():
            return function(x=x, batch=batch)
        return checkpoint(
            lambda x_arg, batch_arg: function(x=x_arg, batch=batch_arg),
            x,
            batch,
            use_reentrant=False,
        )

    def graph(
        self,
        function: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
    ) -> Tensor:
        if not self._enabled():
            return function(x, edge_index, edge_weight)
        if edge_weight is None:
            return checkpoint(
                lambda x_arg, edge_index_arg: function(x_arg, edge_index_arg, None),
                x,
                edge_index,
                use_reentrant=False,
            )
        return checkpoint(
            function,
            x,
            edge_index,
            edge_weight,
            use_reentrant=False,
        )

    def pool(
        self,
        function: Callable[[Tensor, Tensor, Tensor], PoolOutput],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> PoolOutput:
        if not self._enabled():
            return function(x, edge_index, batch)

        values = checkpoint(
            lambda x_arg, edge_index_arg, batch_arg: self._pool_tuple(
                function,
                x_arg,
                edge_index_arg,
                batch_arg,
            ),
            x,
            edge_index,
            batch,
            use_reentrant=False,
        )
        return PoolOutput(
            x=values[0],
            edge_index=values[1],
            batch=values[2],
            edge_weight=values[3],
            aux_loss=values[4],
        )

    @staticmethod
    def _pool_tuple(
        function: Callable[[Tensor, Tensor, Tensor], PoolOutput],
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        output = function(x, edge_index, batch)
        return (
            output.x,
            output.edge_index,
            output.batch,
            output.edge_weight,
            output.aux_loss,
        )
