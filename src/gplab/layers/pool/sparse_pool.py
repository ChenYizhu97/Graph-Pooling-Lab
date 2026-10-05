"""Feature-affine selection with TGP's native selection, reduction, and connection pipeline."""
from collections.abc import Callable

from tgp.poolers import TopkPooling
from tgp.select import Select, SelectOutput, TopkSelect
from torch import Tensor
from torch.nn import Linear

from .tgp_connect import SelectionConnect


class AffineSelect(Select):
    """Score nodes with activation(Wx + b), without TopK's projection normalization."""
    def __init__(self, in_channels: int, ratio: float | int, nonlinearity: str | Callable) -> None:
        super().__init__()
        self.linear = Linear(in_channels, 1)
        self.topk = TopkSelect(ratio=ratio, act=nonlinearity)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()
        self.topk.reset_parameters()

    def forward(self, x: Tensor, *, batch: Tensor | None = None, **kwargs) -> SelectOutput:
        return self.topk(self.linear(x), batch=batch)


class SparsePooling(TopkPooling):
    """Use GPLab's affine scorer while reusing TGP's weighted feature and induced-edge pooling."""
    def __init__(self, in_channels: int, ratio: float | int = 0.5,
                 nonlinearity: str | Callable = "tanh") -> None:
        # Width one creates a parameter-free placeholder selector, replaced below.
        super().__init__(in_channels=1, ratio=ratio, nonlinearity=nonlinearity, remove_self_loops=False)
        self.selector = AffineSelect(in_channels, ratio, nonlinearity)
        self.connector = SelectionConnect(remove_self_loops=False)
        self.reset_parameters()
