from abc import ABC, abstractmethod
import inspect
from typing import Optional

import toml
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MLP, BatchNorm, LayerNorm
from torch_geometric.nn.resolver import activation_resolver
from torch.utils.checkpoint import checkpoint

from gplab.layers.functional import readout
from gplab.layers.pool.contracts import PoolOutput, validate_pool_output
from gplab.layers.resolver import conv_resolver, pool_resolver
from gplab.paths import default_config_path

DEFAULT_CONF = default_config_path("model.toml")


class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pool_validated = False

    def forward(self, data: Data):
        raise NotImplementedError

    def _pool(
        self,
        pool: callable,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> PoolOutput:
        if pool is None:
            return PoolOutput(x=x, edge_index=edge_index, batch=batch)

        pool_out = pool(x=x, edge_index=edge_index, batch=batch)

        if not self._pool_validated:
            validate_pool_output(pool_out, pool.__class__.__name__)
            self._pool_validated = True

        return pool_out

    def _load_from_config(self, config: dict) -> None:
        self.p_dropout = config["p_dropout"]
        self.hidden_features = config["hidden_features"]
        self.nonlinearity = config["nonlinearity"]

        self.pre_gnn = [self.n_node_features, *list(config["pre_gnn"])]
        self.post_gnn = [*list(config["post_gnn"]), self.n_classes]
        self.CONV = config["conv_layer"]


class GraphClassifierBase(BaseModel, ABC):
    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        pool_method: Optional[str] = None,
        ratio: float = 0.5,
        config: Optional[dict] = None,
        avg_node_num: Optional[float] = None,
        activation_checkpoint: bool = False,
        norm: str = "layer_norm",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        self.pool_method = pool_method
        self.norm_name = norm
        self.activation_checkpoint = activation_checkpoint

        if config is None:
            print("No config provided to model...Using default config...")
            config = toml.load(DEFAULT_CONF)["model"]

        self._load_from_config(config)

        self.nonlinearity = activation_resolver(self.nonlinearity)
        self.conv_layer = conv_resolver(self.CONV)
        self.norm_layer = self._resolve_norm(norm)

        self.pre_gnn = self._build_pre_gnn()
        self.pool = pool_resolver(
            self.pool_method,
            self.hidden_features,
            ratio=ratio,
            avg_node_num=avg_node_num,
            nonlinearity=self.nonlinearity,
        )
        self.conv1 = self.conv_layer(self.hidden_features, self.hidden_features)
        self.conv1_supports_edge_weight = self._conv_supports_edge_weight(self.conv1)
        self.ln_conv1 = self.norm_layer(self.hidden_features)
        self.conv2 = self.conv_layer(self.hidden_features, self.hidden_features)
        self.conv2_supports_edge_weight = self._conv_supports_edge_weight(self.conv2)
        self.ln_conv2 = self.norm_layer(self.hidden_features)
        self.global_pool = readout
        self.post_gnn = self._build_post_gnn()

        self.reset_parameters()

    def forward(self, data: Data) -> tuple[Tensor, Optional[Tensor]]:
        x, edge_index, batch, edge_weight = self._unpack_graph(data)

        x = self._checkpoint_tensor(self.pre_gnn, x)
        x = self._apply_checkpointed_conv_block(
            self.conv1,
            self.ln_conv1,
            x,
            edge_index,
            edge_weight=edge_weight,
            supports_edge_weight=self.conv1_supports_edge_weight,
        )

        before_pool = self._readout_before_pool(x, batch)
        pool_out = self._checkpoint_pool(x, edge_index, batch)

        x = self._apply_checkpointed_conv_block(
            self.conv2,
            self.ln_conv2,
            pool_out.x,
            pool_out.edge_index,
            edge_weight=pool_out.edge_weight,
            supports_edge_weight=self.conv2_supports_edge_weight,
        )
        after_pool = self._checkpoint_tensor_batch(self.global_pool, x, pool_out.batch)

        graph_embedding = self._merge_graph_embeddings(before_pool, after_pool)
        logits = self._checkpoint_tensor(self.post_gnn, graph_embedding)
        y = F.log_softmax(logits, dim=1)

        return y, pool_out.aux_loss

    def reset_parameters(self) -> None:
        self.pre_gnn.reset_parameters()
        if self.pool is not None:
            self.pool.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post_gnn.reset_parameters()
        self.ln_conv1.reset_parameters()
        self.ln_conv2.reset_parameters()

    def _resolve_norm(self, norm: str):
        if norm == "layer_norm":
            return LayerNorm
        if norm == "batch_norm":
            return BatchNorm
        raise ValueError(f"Unsupported norm '{norm}'. Use 'layer_norm' or 'batch_norm'.")

    def _build_pre_gnn(self) -> MLP:
        return MLP(
            channel_list=self.pre_gnn,
            act=self.nonlinearity,
            norm=self.norm_name,
            bias=True,
            plain_last=False,
            dropout=self.p_dropout,
        )

    def _build_post_gnn(self) -> MLP:
        bias = [True] * (len(self.post_gnn) - 2) + [False]
        return MLP(
            channel_list=self.post_gnn,
            act=self.nonlinearity,
            norm=self.norm_name,
            bias=bias,
            plain_last=True,
            dropout=self.p_dropout,
        )

    def _unpack_graph(self, data: Data) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        edge_weight = getattr(data, "edge_weight", None)
        return data.x, data.edge_index, batch, edge_weight

    def _should_checkpoint(self) -> bool:
        return self.activation_checkpoint and torch.is_grad_enabled()

    def _checkpoint_tensor(self, fn: callable, x: Tensor) -> Tensor:
        if not self._should_checkpoint():
            return fn(x)
        return checkpoint(fn, x, use_reentrant=False)

    def _checkpoint_tensor_batch(self, fn: callable, x: Tensor, batch: Tensor) -> Tensor:
        if not self._should_checkpoint():
            return fn(x=x, batch=batch)
        return checkpoint(lambda x_arg, batch_arg: fn(x=x_arg, batch=batch_arg), x, batch, use_reentrant=False)

    def _checkpoint_pool(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> PoolOutput:
        if not self._should_checkpoint():
            return self._pool(self.pool, x=x, edge_index=edge_index, batch=batch)

        pool_x, pool_edge_index, pool_batch, pool_edge_weight, pool_aux_loss = checkpoint(
            self._pool_for_checkpoint,
            x,
            edge_index,
            batch,
            use_reentrant=False,
        )
        return PoolOutput(
            x=pool_x,
            edge_index=pool_edge_index,
            batch=pool_batch,
            edge_weight=pool_edge_weight,
            aux_loss=pool_aux_loss,
        )

    def _pool_for_checkpoint(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        pool_out = self._pool(self.pool, x=x, edge_index=edge_index, batch=batch)
        return pool_out.x, pool_out.edge_index, pool_out.batch, pool_out.edge_weight, pool_out.aux_loss

    def _apply_checkpointed_conv_block(
        self,
        conv: torch.nn.Module,
        norm: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        supports_edge_weight: bool = False,
    ) -> Tensor:
        if not self._should_checkpoint():
            return self._apply_conv_block(
                conv,
                norm,
                x,
                edge_index,
                edge_weight=edge_weight,
                supports_edge_weight=supports_edge_weight,
            )

        if edge_weight is None:
            return checkpoint(
                lambda x_arg, edge_index_arg: self._apply_conv_block(
                    conv,
                    norm,
                    x_arg,
                    edge_index_arg,
                    edge_weight=None,
                    supports_edge_weight=supports_edge_weight,
                ),
                x,
                edge_index,
                use_reentrant=False,
            )

        return checkpoint(
            lambda x_arg, edge_index_arg, edge_weight_arg: self._apply_conv_block(
                conv,
                norm,
                x_arg,
                edge_index_arg,
                edge_weight=edge_weight_arg,
                supports_edge_weight=supports_edge_weight,
            ),
            x,
            edge_index,
            edge_weight,
            use_reentrant=False,
        )

    def _apply_conv_block(
        self,
        conv: torch.nn.Module,
        norm: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        supports_edge_weight: bool = False,
    ) -> Tensor:
        if edge_weight is not None and supports_edge_weight:
            x = conv(x, edge_index, edge_weight=edge_weight)
        else:
            if edge_weight is not None:
                edge_index = self._filter_zero_weight_edges(edge_index, edge_weight)
            x = conv(x, edge_index)
        x = norm(x)
        return self.nonlinearity(x)

    @staticmethod
    def _conv_supports_edge_weight(conv: torch.nn.Module) -> bool:
        params = inspect.signature(conv.forward).parameters
        return "edge_weight" in params

    @staticmethod
    def _filter_zero_weight_edges(edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        keep_mask = edge_weight != 0
        if bool(keep_mask.all()):
            return edge_index
        return edge_index[:, keep_mask]

    def _readout_before_pool(self, x: Tensor, batch: Tensor) -> Optional[Tensor]:
        return None

    @abstractmethod
    def _merge_graph_embeddings(
        self,
        before_pool: Optional[Tensor],
        after_pool: Tensor,
    ) -> Tensor:
        raise NotImplementedError
