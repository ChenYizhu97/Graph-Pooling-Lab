from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, LayerNorm, MLP
from torch_geometric.nn.resolver import activation_resolver
from torch.utils.checkpoint import checkpoint

from gplab.benchmark.case import ModelConfig
from gplab.graph import ConnectivityType
from gplab.layers.conv.profiles import CONV_PROFILES
from gplab.layers.functional import readout
from gplab.layers.pool.profiles import load_pooling_profile
from gplab.layers.pool.pooling_output import PoolingOutput, validate_pooling_output


class GraphClassifier(torch.nn.Module):
    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        config: ModelConfig,
        pool_method: str,
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
        self.variant = config.variant
        self.activation_checkpoint = activation_checkpoint
        self._pool_validated = False

        pre_conv_profile = CONV_PROFILES[config.pre_conv]
        post_conv_profile = CONV_PROFILES[config.post_conv]
        pool_profile = load_pooling_profile(pool_method)
        norm_layer = self._resolve_norm(norm)
        self.pre_gnn = self._build_pre_gnn(config)
        self.pool_module = pool_profile.build(
            in_channels=self.hidden_features,
            ratio=ratio,
            avg_node_num=avg_node_num,
            nonlinearity=pool_nonlinearity,
        )
        self.pre_conv = pre_conv_profile.build(
            self.hidden_features,
            self.hidden_features,
        )
        self.pre_conv_supports_edge_weight = pre_conv_profile.supports(
            ConnectivityType.SCALAR
        )
        self.pre_norm = norm_layer(self.hidden_features)
        self.post_conv = post_conv_profile.build(
            self.hidden_features,
            self.hidden_features,
        )
        self.post_conv_supports_edge_weight = post_conv_profile.supports(
            ConnectivityType.SCALAR
        )
        self.post_norm = norm_layer(self.hidden_features)
        self.post_gnn = self._build_post_gnn(config)
        self.reset_parameters()

    def forward(self, data: Data) -> tuple[Tensor, Optional[Tensor]]:
        x, edge_index, batch, edge_weight = self._unpack_graph(data)
        if self.activation_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                self._forward_backbone,
                x,
                edge_index,
                batch,
                edge_weight,
                use_reentrant=False,
            )
        return self._forward_backbone(x, edge_index, batch, edge_weight)

    def _forward_backbone(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = self.pre_gnn(x)
        if edge_weight is None:
            x = self.pre_conv(x, edge_index)
        elif self.pre_conv_supports_edge_weight:
            x = self.pre_conv(x, edge_index, edge_weight=edge_weight)
        else:
            raise ValueError(
                "Scalar connectivity reached a pre-pooling convolution "
                "that cannot consume edge_weight."
            )
        x = self.nonlinearity(self.pre_norm(x))

        before_pool = readout(x=x, batch=batch) if self.variant == "sum" else None
        pool_output = self._apply_pool(x, edge_index, batch, edge_weight)

        if pool_output.edge_weight is None:
            x = self.post_conv(pool_output.x, pool_output.edge_index)
        elif self.post_conv_supports_edge_weight:
            x = self.post_conv(
                pool_output.x,
                pool_output.edge_index,
                edge_weight=pool_output.edge_weight,
            )
        else:
            raise ValueError(
                "Scalar connectivity reached a post-pooling convolution "
                "that cannot consume edge_weight."
            )
        x = self.nonlinearity(self.post_norm(x))

        graph_embedding = readout(x=x, batch=pool_output.batch)
        if before_pool is not None:
            graph_embedding = before_pool + graph_embedding
        logits = self.post_gnn(graph_embedding)
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
        self.pre_conv.reset_parameters()
        self.post_conv.reset_parameters()
        self.post_gnn.reset_parameters()
        self.pre_norm.reset_parameters()
        self.post_norm.reset_parameters()

    def _apply_pool(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Optional[Tensor],
    ) -> PoolingOutput:
        if self.pool_module is None:
            return PoolingOutput(
                x=x,
                edge_index=edge_index,
                batch=batch,
                edge_weight=edge_weight,
            )
        output = self.pool_module(x=x, edge_index=edge_index, batch=batch, edge_weight=edge_weight)
        if not self._pool_validated:
            validate_pooling_output(output, self.pool_module.__class__.__name__)
            self._pool_validated = True
        return output

    def _build_pre_gnn(self, config: ModelConfig) -> MLP:
        return MLP(
            channel_list=[self.n_node_features, *config.pre_gnn],
            act=self.nonlinearity,
            norm=self.norm_name,
            bias=True,
            plain_last=False,
            dropout=self.p_dropout,
        )

    def _build_post_gnn(self, config: ModelConfig) -> MLP:
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
