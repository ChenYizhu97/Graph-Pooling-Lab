from abc import ABC, abstractmethod
import inspect
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, LayerNorm, MLP
from torch_geometric.nn.resolver import activation_resolver
from torch.utils.checkpoint import checkpoint

from gplab.experiment.spec import ModelSpec
from gplab.layers.functional import readout
from gplab.layers.pool.contracts import PoolOutput, validate_pool_output
from gplab.layers.resolver import conv_resolver, pool_resolver


class GraphClassifierBase(torch.nn.Module, ABC):
    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        config: ModelSpec,
        pool_method: Optional[str] = None,
        ratio: float = 0.5,
        pool_nonlinearity: str = "tanh",
        avg_node_num: Optional[float] = None,
        activation_checkpoint: bool = False,
        norm: str = "layer_norm",
    ) -> None:
        super().__init__()
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        self.hidden_features = config.hidden_features
        self.p_dropout = config.p_dropout
        self.nonlinearity = activation_resolver(config.nonlinearity)
        self.norm_name = norm
        self.activation_checkpoint = activation_checkpoint
        self._pool_validated = False

        conv_layer = conv_resolver(config.conv_layer)
        norm_layer = self._resolve_norm(norm)
        self.pre_gnn = self._build_pre_gnn(config)
        self.pool_module = pool_resolver(
            pool_method,
            self.hidden_features,
            ratio=ratio,
            avg_node_num=avg_node_num,
            nonlinearity=pool_nonlinearity,
        )
        self.conv1 = conv_layer(self.hidden_features, self.hidden_features)
        self.conv1_supports_edge_weight = self._conv_supports_edge_weight(self.conv1)
        self.norm1 = norm_layer(self.hidden_features)
        self.conv2 = conv_layer(self.hidden_features, self.hidden_features)
        self.conv2_supports_edge_weight = self._conv_supports_edge_weight(self.conv2)
        self.norm2 = norm_layer(self.hidden_features)
        self.readout = readout
        self.post_gnn = self._build_post_gnn(config)
        self.reset_parameters()

    def forward(self, data: Data) -> tuple[Tensor, Optional[Tensor]]:
        x, edge_index, batch, edge_weight = self._unpack_graph(data)
        x = self._run_with_optional_checkpoint(self.pre_gnn, x)
        x = self._execute_conv_block(
            self.conv1,
            self.norm1,
            x,
            edge_index,
            edge_weight,
            self.conv1_supports_edge_weight,
        )

        before_pool = self._readout_before_pool(x, batch)
        pool_output = self._pool_with_optional_checkpoint(x, edge_index, batch)
        x = self._execute_conv_block(
            self.conv2,
            self.norm2,
            pool_output.x,
            pool_output.edge_index,
            pool_output.edge_weight,
            self.conv2_supports_edge_weight,
        )
        after_pool = self._readout_with_optional_checkpoint(x, pool_output.batch)
        graph_embedding = self._merge_graph_embeddings(before_pool, after_pool)
        logits = self._run_with_optional_checkpoint(self.post_gnn, graph_embedding)
        return F.log_softmax(logits, dim=1), pool_output.aux_loss

    def reset_parameters(self) -> None:
        self.pre_gnn.reset_parameters()
        if self.pool_module is not None:
            reset_pool = getattr(self.pool_module, "reset_parameters", None)
            if not callable(reset_pool):
                raise TypeError(
                    f"Pooling module '{self.pool_module.__class__.__name__}' "
                    "must implement reset_parameters() "
                    "because GPLab reuses the model across seeded runs."
                )
            reset_pool()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post_gnn.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def _apply_pool(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> PoolOutput:
        if self.pool_module is None:
            return PoolOutput(x=x, edge_index=edge_index, batch=batch)
        output = self.pool_module(x=x, edge_index=edge_index, batch=batch)
        if not self._pool_validated:
            validate_pool_output(output, self.pool_module.__class__.__name__)
            self._pool_validated = True
        return output

    def _should_checkpoint_activations(self) -> bool:
        return self.activation_checkpoint and torch.is_grad_enabled()

    def _run_with_optional_checkpoint(self, function: Callable, *args):
        if not self._should_checkpoint_activations():
            return function(*args)
        return checkpoint(function, *args, use_reentrant=False)

    def _readout_with_optional_checkpoint(self, x: Tensor, batch: Tensor) -> Tensor:
        return self._run_with_optional_checkpoint(
            lambda x_arg, batch_arg: self.readout(x=x_arg, batch=batch_arg),
            x,
            batch,
        )

    def _pool_with_optional_checkpoint(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> PoolOutput:
        if not self._should_checkpoint_activations():
            return self._apply_pool(x, edge_index, batch)

        pool_values = self._run_with_optional_checkpoint(
            lambda x_arg, edge_index_arg, batch_arg: self._pool_checkpoint_values(
                x_arg,
                edge_index_arg,
                batch_arg,
            ),
            x,
            edge_index,
            batch,
        )
        (
            pooled_x,
            pooled_edge_index,
            pooled_batch,
            pooled_edge_weight,
            auxiliary_loss,
        ) = pool_values
        return PoolOutput(
            x=pooled_x,
            edge_index=pooled_edge_index,
            batch=pooled_batch,
            edge_weight=pooled_edge_weight,
            aux_loss=auxiliary_loss,
        )

    def _pool_checkpoint_values(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        output = self._apply_pool(x, edge_index, batch)
        return output.x, output.edge_index, output.batch, output.edge_weight, output.aux_loss

    def _execute_conv_block(
        self,
        conv: torch.nn.Module,
        norm: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        supports_edge_weight: bool,
    ) -> Tensor:
        if edge_weight is None:
            return self._run_with_optional_checkpoint(
                lambda x_arg, edge_index_arg: self._apply_conv_block(
                    conv,
                    norm,
                    x_arg,
                    edge_index_arg,
                    None,
                    supports_edge_weight,
                ),
                x,
                edge_index,
            )

        return self._run_with_optional_checkpoint(
            lambda x_arg, edge_index_arg, edge_weight_arg: self._apply_conv_block(
                conv,
                norm,
                x_arg,
                edge_index_arg,
                edge_weight_arg,
                supports_edge_weight,
            ),
            x,
            edge_index,
            edge_weight,
        )

    def _apply_conv_block(
        self,
        conv: torch.nn.Module,
        norm: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        supports_edge_weight: bool,
    ) -> Tensor:
        if edge_weight is not None and supports_edge_weight:
            x = conv(x, edge_index, edge_weight=edge_weight)
        else:
            if edge_weight is not None:
                edge_index = self._filter_zero_weight_edges(edge_index, edge_weight)
            x = conv(x, edge_index)
        return self.nonlinearity(norm(x))

    def _build_pre_gnn(self, config: ModelSpec) -> MLP:
        return MLP(
            channel_list=[self.n_node_features, *config.pre_gnn],
            act=self.nonlinearity,
            norm=self.norm_name,
            bias=True,
            plain_last=False,
            dropout=self.p_dropout,
        )

    def _build_post_gnn(self, config: ModelSpec) -> MLP:
        channels = [*config.post_gnn, self.n_classes]
        bias = [True] * (len(channels) - 2) + [False]
        return MLP(
            channel_list=channels,
            act=self.nonlinearity,
            norm=self.norm_name,
            bias=bias,
            plain_last=True,
            dropout=self.p_dropout,
        )

    @staticmethod
    def _resolve_norm(norm: str):
        if norm == "layer_norm":
            return LayerNorm
        if norm == "batch_norm":
            return BatchNorm
        raise ValueError(f"Unsupported norm '{norm}'. Use 'layer_norm' or 'batch_norm'.")

    @staticmethod
    def _unpack_graph(data: Data) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        return data.x, data.edge_index, batch, getattr(data, "edge_weight", None)

    @staticmethod
    def _conv_supports_edge_weight(conv: torch.nn.Module) -> bool:
        return "edge_weight" in inspect.signature(conv.forward).parameters

    @staticmethod
    def _filter_zero_weight_edges(edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        keep_mask = edge_weight != 0
        return edge_index if bool(keep_mask.all()) else edge_index[:, keep_mask]

    def _readout_before_pool(self, x: Tensor, batch: Tensor) -> Optional[Tensor]:
        return None

    @abstractmethod
    def _merge_graph_embeddings(
        self,
        before_pool: Optional[Tensor],
        after_pool: Tensor,
    ) -> Tensor:
        raise NotImplementedError
