"""Verify GPLab's configured TGP migration preserves its previous dense operator semantics."""
import unittest

import torch
from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from gplab.layers.pool import load_pooling_profile


class DensePoolingMigrationTests(unittest.TestCase):
    def test_configured_outputs_losses_and_gradients_match_previous_pipeline(self):
        # Unequal sizes expose padding/loss normalization differences. K exceeds
        # the smaller graph size so every cluster slot must survive conversion.
        batch = torch.tensor([0, 0, 0, 1, 1])
        edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        weights = torch.tensor([.2, .2, .8, .8, .7, .7], dtype=torch.float64)
        for name in ("densepool", "diffpool", "mincutpool"):
            with self.subTest(pool=name):
                torch.manual_seed(7)
                x = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
                pool = load_pooling_profile(name).build(in_channels=2, k=4).double()
                actual = pool(x=x, adj=edges, batch=batch, edge_weight=weights)
                dense_x, mask = to_dense_batch(x, batch)
                adjacency = to_dense_adj(edges, batch, edge_attr=weights)
                logits = (pool.selector.gnn(dense_x, adjacency, mask) if name == "diffpool"
                          else pool.selector.mlp(dense_x))
                if name == "diffpool":
                    expected_x, expected_adj, link, entropy = dense_diff_pool(dense_x, adjacency, logits, mask)
                    expected_losses = {"link_loss": .1 * link, "entropy_loss": .1 * entropy}
                elif name == "mincutpool":
                    expected_x, expected_adj, cut, ortho = dense_mincut_pool(dense_x, adjacency, logits, mask)
                    expected_losses = {"cut_loss": .5 * cut, "ortho_loss": ortho}
                else:
                    assignment = logits.softmax(-1) * mask.unsqueeze(-1)
                    expected_x = assignment.transpose(1, 2) @ dense_x
                    expected_adj = assignment.transpose(1, 2) @ adjacency @ assignment
                    expected_losses = {}
                torch.testing.assert_close(actual.x, expected_x.flatten(0, 1), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(actual.edge_weight, expected_adj.flatten(), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(to_dense_adj(actual.edge_index, actual.batch, actual.edge_weight),
                                           expected_adj, atol=1e-6, rtol=1e-6)
                self.assertEqual(torch.bincount(actual.batch).tolist(), [4, 4])
                self.assertEqual(actual.edge_index.size(1), 2 * 4 * 4)
                self.assertEqual(set(actual.loss or {}), set(expected_losses))
                for key, value in expected_losses.items():
                    torch.testing.assert_close(actual.loss[key], value, atol=1e-6, rtol=1e-6)
                actual_objective = actual.x.square().sum() + actual.edge_weight.sum() + sum((actual.loss or {}).values())
                expected_objective = expected_x.square().sum() + expected_adj.sum() + sum(expected_losses.values())
                inputs = (x, *pool.parameters())
                gradients = torch.autograd.grad(actual_objective, inputs, retain_graph=True)
                expected_gradients = torch.autograd.grad(expected_objective, inputs)
                for gradient, expected in zip(gradients, expected_gradients, strict=True):
                    torch.testing.assert_close(gradient, expected, atol=1e-5, rtol=1e-5)

    def test_empty_cluster_slots_and_zero_edges_remain_in_output(self):
        pool = load_pooling_profile("densepool").build(in_channels=1, k=3)
        with torch.no_grad():
            pool.selector.mlp.lins[0].weight.zero_()
            pool.selector.mlp.lins[0].bias.copy_(torch.tensor([1000., -1000., -1000.]))
        output = pool(x=torch.ones(2, 1), adj=torch.tensor([[0, 1], [1, 0]]),
                      batch=torch.zeros(2, dtype=torch.long))
        self.assertEqual(output.x.size(0), 3)
        self.assertEqual(output.edge_index.size(1), 9)
        self.assertEqual(torch.count_nonzero(output.edge_weight).item(), 1)
