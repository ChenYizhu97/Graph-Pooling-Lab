import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gplab.layers.resolver import pool_resolver


def build_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 1.0],
            [1.0, 1.0, 0.0],
            [0.3, 0.7, 1.0],
        ],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 3],
            [1, 0, 2, 1, 4, 3, 3, 4],
        ],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    return x, edge_index, batch


def inspect_pool(pool_name: str, ratio: float = 0.8) -> None:
    x, edge_index, batch = build_batch()
    avg_node_num = x.size(0) / int(batch.max().item() + 1)
    expected_clusters = max(1, int(avg_node_num * ratio))

    torch.manual_seed(7)
    pool_module = pool_resolver(
        pool_name,
        in_channels=x.size(-1),
        ratio=ratio,
        avg_node_num=avg_node_num,
        nonlinearity="relu",
    )
    pool_module.eval()

    with torch.no_grad():
        pool_output = pool_module(x=x, edge_index=edge_index, batch=batch)

    per_graph_clusters = torch.bincount(pool_output.batch, minlength=2)
    print(f"[{pool_name}]")
    print(f"  input_nodes_per_graph: {torch.bincount(batch, minlength=2).tolist()}")
    print(f"  output_clusters_per_graph: {per_graph_clusters.tolist()}")
    print(f"  expected_fixed_clusters: {expected_clusters}")
    print(f"  pooled_x_shape: {list(pool_output.x.shape)}")
    print(f"  pooled_edge_count: {int(pool_output.edge_index.size(1))}")
    if pool_output.edge_weight is None:
        print("  pooled_edge_weight: None")
    else:
        print(
            "  pooled_edge_weight_stats: "
            f"min={float(pool_output.edge_weight.min()):.6f}, "
            f"max={float(pool_output.edge_weight.max()):.6f}, "
            f"mean={float(pool_output.edge_weight.mean()):.6f}"
        )

    if not torch.all(per_graph_clusters == expected_clusters):
        raise SystemExit(
            f"{pool_name}: expected every graph to keep {expected_clusters} coarse clusters, "
            f"got {per_graph_clusters.tolist()}"
        )
    if pool_output.edge_weight is None:
        raise SystemExit(f"{pool_name}: dense adapter did not preserve coarse edge weights")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dense adapter output semantics.")
    parser.add_argument(
        "--pools",
        nargs="+",
        default=["densepool", "diffpool", "mincutpool"],
        help="Dense pooling methods to inspect.",
    )
    parser.add_argument("--pool-ratio", type=float, default=0.8)
    args = parser.parse_args()

    for pool_name in args.pools:
        inspect_pool(pool_name, ratio=args.pool_ratio)


if __name__ == "__main__":
    main()
