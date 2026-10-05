"""GPLab pooling configuration, output boundaries, and loss integration."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tgp.connect import SparseConnect
from tgp.poolers import SAGPooling, TopkPooling
from tgp.src import PoolingOutput
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from gplab.benchmark.config import ModelConfig
from gplab.layers.pool import POOLING_PROFILES, validate_pooling_output
from gplab.layers.pool import PoolingOutput as ExportedPoolingOutput
from gplab.layers.pool.profiles import PoolingProfile
from gplab.layers.pool.sparse_pool import SparsePooling
from gplab.layers.pool.tgp_connect import SelectionConnect
from gplab.model import GraphClassifier


class PoolingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.graph = Data(x=torch.tensor([[1., .2], [.3, .8], [-.2, .4]]),
                          edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                          batch=torch.zeros(3, dtype=torch.long))
        self.config = ModelConfig(2, "relu", 0., "GCN", "GCN", (2,), (4, 2), "plain")

    def test_sparse_pool_preserves_affine_scores_and_weighted_induced_graph(self):
        pool = SparsePooling(1, ratio=.6, nonlinearity="identity")
        with torch.no_grad():
            pool.selector.linear.weight.fill_(2)
            pool.selector.linear.bias.fill_(1)
        x = torch.tensor([[1.], [3.], [2.], [4.], [1.], [5.]], requires_grad=True)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        edges = torch.tensor([[0, 1, 2, 1, 3, 5, 4], [1, 2, 1, 1, 5, 3, 5]])
        weights = torch.tensor([.1, .2, .3, .4, .5, .6, .7])
        output = pool(x=x, adj=edges, batch=batch, edge_weight=weights)
        # The custom affine score includes bias and has no projection normalization.
        torch.testing.assert_close(output.x, torch.tensor([[21.], [10.], [55.], [36.]]))
        torch.testing.assert_close(output.batch, torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(output.edge_index, torch.tensor([[0, 1, 0, 3, 2], [1, 0, 0, 2, 3]]))
        torch.testing.assert_close(output.edge_weight, weights[[1, 2, 3, 4, 5]])
        self.assertEqual(output.so.num_supernodes, 4)
        output.x.sum().backward()
        self.assertIsNotNone(pool.selector.linear.weight.grad)
        self.assertIsNotNone(pool.selector.linear.bias.grad)

    def test_builtin_outputs_meet_classifier_boundary(self):
        self.assertIs(ExportedPoolingOutput, PoolingOutput)
        for name in POOLING_PROFILES:
            with self.subTest(pool=name):
                model = GraphClassifier(2, 2, self.config, name, avg_node_num=3)
                output = model._apply_pool(self.graph.x, self.graph.edge_index, self.graph.batch, None)
                self.assertIs(type(output), PoolingOutput)
                validate_pooling_output(output, name)

    def test_dense_profiles_configure_native_backend_semantics(self):
        from tgp.poolers import DiffPool, MinCutPooling
        diff = POOLING_PROFILES["diffpool"].build(in_channels=2, k=2)
        cut = POOLING_PROFILES["mincutpool"].build(in_channels=2, k=2)
        dense = POOLING_PROFILES["densepool"].build(in_channels=2, k=2)
        self.assertIsInstance(diff, DiffPool)
        self.assertIsInstance(cut, MinCutPooling)
        self.assertEqual((diff.link_loss_coeff, diff.ent_loss_coeff), (.1, .1))
        self.assertEqual((cut.cut_loss_coeff, cut.ortho_loss_coeff), (.5, 1.))
        self.assertTrue(diff.normalize_loss)
        self.assertFalse(diff.connector.degree_norm)
        self.assertFalse(diff.connector.remove_self_loops)
        self.assertTrue(cut.connector.degree_norm)
        self.assertTrue(cut.connector.remove_self_loops)
        self.assertIsNone(dense.compute_loss(None, None, None))

    def test_classifier_aggregates_loss_terms_with_checkpointing(self):
        class LossFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.penalty = torch.nn.Parameter(torch.tensor(2.))

            def reset_parameters(self):
                pass

            def forward(self, x, adj, *, batch, edge_weight=None):
                return PoolingOutput(x=x, edge_index=adj, batch=batch, edge_weight=edge_weight,
                                     loss={"first": self.penalty.square(), "second": .3 * self.penalty})

        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                profile = PoolingProfile(lambda *args, **kwargs: LossFixture(), POOLING_PROFILES["nopool"].signatures)
                with patch("gplab.model.classifier.load_pooling_profile", return_value=profile):
                    model = GraphClassifier(2, 2, self.config, "custom", activation_checkpoint=checkpoint)
                logits, auxiliary_loss = model(self.graph)
                torch.testing.assert_close(auxiliary_loss, torch.tensor(4.6))
                (logits.square().sum() + auxiliary_loss).backward()
                torch.testing.assert_close(model.pool_module.penalty.grad, torch.tensor(4.3))

    def test_invalid_loss_and_dense_graph_are_rejected_at_sparse_boundary(self):
        for loss in (torch.tensor(1.), {"bad": torch.ones(2)}, {"bad": None}):
            with self.subTest(loss=loss), self.assertRaises((TypeError, ValueError)):
                validate_pooling_output(PoolingOutput(x=self.graph.x, edge_index=self.graph.edge_index,
                                                      batch=self.graph.batch, loss=loss), "custom")
        with self.assertRaisesRegex(ValueError, "x must have shape"):
            validate_pooling_output(PoolingOutput(x=self.graph.x[None],
                                                  edge_index=self.graph.edge_index,
                                                  batch=self.graph.batch), "custom")

    def test_profiles_apply_gplab_topk_and_sag_configuration(self):
        for channels in (1, 2):
            topk = POOLING_PROFILES["topkpool"].build(
                in_channels=channels, ratio=.7, avg_node_num=None, nonlinearity="relu",
            )
            self.assertIs(type(topk), TopkPooling)
            self.assertEqual(topk.selector.weight.shape, (1, channels))
            self.assertEqual(topk.selector.ratio, .7)
            self.assertIsInstance(topk.selector.act, torch.nn.ReLU)
            self.assertIsInstance(topk.connector, SelectionConnect)
            self.assertFalse(topk.connector.remove_self_loops)
        sag = POOLING_PROFILES["sagpool"].build(
            in_channels=2, ratio=.7, avg_node_num=None,
        )
        self.assertIs(type(sag), SAGPooling)
        self.assertIsInstance(sag.gnn, GCNConv)
        self.assertIsInstance(sag.selector.act, torch.nn.Tanh)
        self.assertEqual(sag.selector.ratio, .7)
        self.assertIsInstance(sag.connector, SelectionConnect)
        self.assertFalse(sag.connector.remove_self_loops)

    def test_native_parameters_reach_pool_constructors(self):
        model = GraphClassifier(2, 2, self.config, "topkpool", pool_params={
            "ratio": 2, "multiplier": 3.0, "nonlinearity": "relu", "remove_self_loops": True,
        })
        self.assertEqual(model.pool_module.multiplier, 3.0)
        self.assertEqual(model.pool_module.selector.ratio, 2)
        self.assertTrue(model.pool_module.connector.remove_self_loops)
        sag = POOLING_PROFILES["sagpool"].build(in_channels=2, nonlinearity="relu", improved=True)
        self.assertIsInstance(sag.selector.act, torch.nn.ReLU)
        self.assertTrue(sag.gnn.improved)
        for name in ("topkpool", "sagpool", "sparsepool", "densepool"):
            with self.subTest(pool=name), self.assertRaises(TypeError):
                POOLING_PROFILES[name].build(in_channels=2, unknown_option=True)

    def test_dense_size_parameters_use_native_k_or_explicit_ratio(self):
        profile = POOLING_PROFILES["densepool"]
        for params, expected in (({"ratio": 0.5}, 5), ({"ratio": 1.0}, 1), ({"ratio": 3}, 3), ({"k": 4}, 4)):
            pool = profile.build(in_channels=2, avg_node_num=10, **params)
            self.assertEqual(pool.selector.k, expected)
        with self.assertRaisesRegex(ValueError, "either ratio or k"):
            profile.build(in_channels=2, avg_node_num=10, ratio=0.5, k=4)

    def test_connector_corrects_only_partial_selection_indices(self):
        # Mock the backend to isolate GPLab's source-to-cluster index correction.
        edges = torch.tensor([[0, 1], [1, 0]])
        weights = torch.tensor([.2, .4])
        connector = SelectionConnect(remove_self_loops=False)
        for node_count in (2, 3):
            with self.subTest(node_count=node_count):
                selection = SimpleNamespace(num_nodes=node_count, num_supernodes=2,
                                            cluster_index=torch.tensor([1, 0]))
                with patch.object(SparseConnect, "forward", return_value=(edges, weights)) as backend:
                    actual_edges, actual_weights = connector(edges, selection, edge_weight=weights)
                backend.assert_called_once_with(edges, selection, edge_weight=weights)
                expected = torch.tensor([[1, 0], [0, 1]]) if node_count == 3 else edges
                torch.testing.assert_close(actual_edges, expected)
                self.assertIs(actual_weights, weights)
