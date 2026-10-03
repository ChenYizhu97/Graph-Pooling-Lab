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
from gplab.layers.pool.tgp_connect import SelectionConnect
from gplab.model import GraphClassifier


class PoolingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.graph = Data(x=torch.tensor([[1., .2], [.3, .8], [-.2, .4]]),
                          edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                          batch=torch.zeros(3, dtype=torch.long))
        self.config = ModelConfig(2, "relu", 0., "GCN", "GCN", (2,), (4, 2), "plain")

    def test_builtin_outputs_meet_classifier_boundary(self):
        self.assertIs(ExportedPoolingOutput, PoolingOutput)
        for name in POOLING_PROFILES:
            with self.subTest(pool=name):
                model = GraphClassifier(2, 2, self.config, name, avg_node_num=3)
                output = model._apply_pool(self.graph.x, self.graph.edge_index, self.graph.batch, None)
                self.assertIs(type(output), PoolingOutput)
                validate_pooling_output(output, name)

    def test_dense_loss_dictionary_preserves_original_weights(self):
        # Treat backend results as given; test only GPLab's loss coefficients.
        for name, function, expected in (
            ("diffpool", "dense_diff_pool", {"link": .2, "entropy": .4}),
            ("mincutpool", "dense_mincut_pool", {"mincut": 1., "orthogonality": 4.}),
        ):
            with self.subTest(pool=name):
                pool = POOLING_PROFILES[name].build(in_channels=2, ratio=.7,
                                                   avg_node_num=3, nonlinearity="tanh")
                backend_output = (torch.ones(1, 2, 2), torch.ones(1, 2, 2),
                                  torch.tensor(2.), torch.tensor(4.))
                with patch(f"gplab.layers.pool.dense_pool_adapter.{function}", return_value=backend_output):
                    output = pool(x=self.graph.x, adj=self.graph.edge_index, batch=self.graph.batch)
                self.assertEqual(output.loss.keys(), expected.keys())
                for key in expected:
                    torch.testing.assert_close(output.loss[key], torch.tensor(expected[key]))

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
                profile = PoolingProfile(lambda *args: LossFixture(), POOLING_PROFILES["nopool"].signatures)
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
            in_channels=2, ratio=.7, avg_node_num=None, nonlinearity="relu",
        )
        self.assertIs(type(sag), SAGPooling)
        self.assertIsInstance(sag.gnn, GCNConv)
        self.assertIsInstance(sag.selector.act, torch.nn.Tanh)
        self.assertEqual(sag.selector.ratio, .7)
        self.assertIsInstance(sag.connector, SelectionConnect)
        self.assertFalse(sag.connector.remove_self_loops)

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
