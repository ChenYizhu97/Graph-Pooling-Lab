import inspect
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from gplab.benchmark.case import BenchmarkCase
from gplab.benchmark.comparability import (
    comparable_pools,
    resolve_dataset_connectivity_type,
    validate_comparability,
)
from gplab.benchmark.execution import ExecutionOptions
from gplab.benchmark.plan import RunPlan
from gplab.benchmark.request import BenchmarkRequest
from gplab.experiment.execute import prepare_run
from gplab.data.profiles import DATASET_PROFILES
from gplab.graph import ConnectivityType
from gplab.layers.conv.profiles import CONV_PROFILES
from gplab.layers.functional import readout as graph_readout
from gplab.layers.pool.dense_pool_adapter import DensePoolAdapter
from gplab.layers.pool.profiles import (
    POOLING_PROFILES,
    PoolingSignature,
    load_pooling_profile,
)
from gplab.layers.pool.pooling_output import PoolingOutput
from gplab.model import GraphClassifier
from gplab.jobs.schema import JobSchemaError, normalize_job_shape


class _Dataset(list):
    num_node_features = 2
    num_classes = 2
    connectivity_type = "binary"


def _case(pool="nopool", pre_conv="GCN", post_conv="GCN", variant="plain"):
    return BenchmarkCase.from_mapping({
        "dataset": "MUTAG",
        "pool": {"name": pool, "ratio": 0.5, "nonlinearity": "tanh"},
        "model": {
            "hidden_features": 4, "nonlinearity": "relu", "p_dropout": 0.0,
            "pre_conv": pre_conv, "post_conv": post_conv,
            "pre_gnn": [4], "post_gnn": [8, 4], "variant": variant,
        },
        "training": {
            "runs": 1, "lr": 0.001, "batch_size": 2, "patience": 0, "epochs": 1,
            "split": {"train": 0.5, "val": 0.25},
            "seeds": {"mode": "list", "base": 1, "values": [1], "allow_duplicates": False},
        },
    })


class ComparabilityTests(unittest.TestCase):
    def test_binary_semantics_are_not_inferred_from_unit_weight_tensor(self):
        dataset = _Dataset([
            Data(
                x=torch.ones(2, 2),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_weight=torch.ones(2),
            )
        ])
        self.assertEqual(resolve_dataset_connectivity_type(dataset), ConnectivityType.BINARY)

    def test_scalar_semantics_require_explicit_metadata_and_values(self):
        dataset = _Dataset([
            Data(
                x=torch.ones(2, 2),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_weight=torch.tensor([0.25, 0.75]),
            )
        ])
        dataset.connectivity_type = "scalar"
        self.assertEqual(resolve_dataset_connectivity_type(dataset), ConnectivityType.SCALAR)

    def test_valid_binary_and_scalar_paths(self):
        self.assertEqual(
            validate_comparability(dataset_type=ConnectivityType.BINARY, pool_name="topkpool",
                                   pre_conv="GCN", post_conv="GIN").output_type,
            ConnectivityType.BINARY,
        )
        self.assertEqual(
            validate_comparability(dataset_type=ConnectivityType.BINARY, pool_name="diffpool",
                                   pre_conv="GCN", post_conv="GraphConv").output_type,
            ConnectivityType.SCALAR,
        )

    def test_comparable_pool_resolution_matches_validation(self):
        self.assertEqual(
            comparable_pools(
                dataset_type=ConnectivityType.BINARY,
                pre_conv="GCN",
                post_conv="GIN",
            ),
            ("nopool", "topkpool", "sagpool", "sparsepool"),
        )

    def test_pool_profiles_are_the_signature_and_construction_source(self):
        self.assertEqual(
            POOLING_PROFILES["asapool"].signatures,
            (PoolingSignature(ConnectivityType.BINARY, ConnectivityType.SCALAR),),
        )

    def test_custom_pool_profile_loads_validates_and_builds(self):
        profile_name = "examples.custom_pool_plugin:CUSTOM_POOL_PROFILE"
        profile = load_pooling_profile(profile_name)
        self.assertEqual(
            validate_comparability(
                dataset_type=ConnectivityType.BINARY,
                pool_name=profile_name,
                pre_conv="GCN",
                post_conv="GIN",
            ).output_type,
            ConnectivityType.BINARY,
        )

        pool = profile.build(
            in_channels=2,
            ratio=0.5,
            avg_node_num=None,
            nonlinearity="tanh",
        )
        graph = Data(
            x=torch.randn(4, 2),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
            ),
            batch=torch.zeros(4, dtype=torch.long),
        )
        output = pool(**graph.to_dict())
        self.assertIsInstance(output, PoolingOutput)

        model = GraphClassifier(
            2,
            2,
            _case(profile_name).model,
            pool_method=profile_name,
            ratio=0.5,
            avg_node_num=4,
        )
        self.assertEqual(tuple(model(graph)[0].shape), (1, 2))

    def test_bare_custom_pool_factory_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a PoolingProfile"):
            load_pooling_profile("examples.custom_pool_plugin:_build_custom_pool")

    def test_rejects_unsupported_pool_input_and_downstream(self):
        with self.assertRaisesRegex(ValueError, "cannot consume scalar edge values"):
            validate_comparability(dataset_type=ConnectivityType.BINARY, pool_name="diffpool",
                                   pre_conv="GCN", post_conv="GIN")
        with self.assertRaisesRegex(ValueError, "not declared valid"):
            validate_comparability(dataset_type=ConnectivityType.SCALAR, pool_name="diffpool",
                                   pre_conv="GCN", post_conv="GCN")

    def test_asap_weighted_input_is_rejected_by_current_backend_profile(self):
        with self.assertRaisesRegex(ValueError, "not declared valid"):
            validate_comparability(dataset_type=ConnectivityType.SCALAR, pool_name="asapool",
                                   pre_conv="GCN", post_conv="GCN")

    def test_asap_binary_input_derives_scalar_output(self):
        pool = load_pooling_profile("asapool").build(
            in_channels=2,
            ratio=0.5,
            avg_node_num=None,
            nonlinearity="tanh",
        )
        output = pool(
            x=torch.randn(4, 2),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
            ),
            batch=torch.zeros(4, dtype=torch.long),
        )
        self.assertIsNotNone(output.edge_weight)
        self.assertEqual(output.edge_weight.numel(), output.edge_index.size(1))
        self.assertEqual(
            validate_comparability(dataset_type=ConnectivityType.BINARY, pool_name="asapool",
                                   pre_conv="GCN", post_conv="GCN").output_type,
            ConnectivityType.SCALAR,
        )

    def test_comparability_validation_has_no_universal_pool_ratio(self):
        parameters = inspect.signature(validate_comparability).parameters
        self.assertNotIn("ratio", parameters)
        self.assertNotIn("pool_ratio", parameters)

    def test_legacy_conv_layer_is_rejected(self):
        with self.assertRaisesRegex(JobSchemaError, "Unknown case.model field"):
            normalize_job_shape({
                "case": {
                    "dataset": "MUTAG",
                    "pool": {"name": "nopool", "ratio": 0.5},
                    "model": {"conv_layer": "GraphConv"},
                    "training": {"runs": 1, "epochs": 1, "patience": 0},
                }
            })

    def test_pre_and_post_convolutions_are_independently_validated(self):
        result = validate_comparability(
            dataset_type=ConnectivityType.BINARY,
            pool_name="diffpool",
            pre_conv="GIN",
            post_conv="GCN",
        )
        self.assertEqual(result.output_type, ConnectivityType.SCALAR)

    def test_conv_capabilities_match_convolution_apis(self):
        for name in CONV_PROFILES:
            profile = CONV_PROFILES[name]
            conv = profile.build(4, 4)
            api_accepts_edge_weight = (
                "edge_weight" in inspect.signature(conv.forward).parameters
            )
            self.assertEqual(
                profile.supports(ConnectivityType.SCALAR),
                api_accepts_edge_weight,
            )

    def test_dataset_profiles_are_the_catalog_and_semantics_source(self):
        self.assertIn("MUTAG", DATASET_PROFILES)
        self.assertEqual(
            DATASET_PROFILES["MUTAG"].connectivity_type,
            ConnectivityType.BINARY,
        )

    def test_scalar_pool_output_reaches_post_convolution(self):
        model = GraphClassifier(
            2, 2, _case("diffpool").model, pool_method="diffpool", ratio=0.5, avg_node_num=3,
        )
        seen = {}
        def observe(_module, _args, kwargs, _output):
            seen["edge_weight"] = kwargs.get("edge_weight")

        handle = model.post_conv.register_forward_hook(observe, with_kwargs=True)
        try:
            graph = Data(
                x=torch.randn(3, 2),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                batch=torch.zeros(3, dtype=torch.long),
            )
            model(graph)
        finally:
            handle.remove()
        self.assertIsNotNone(seen.get("edge_weight"))

    def test_pool_output_has_one_scalar_connectivity_channel(self):
        self.assertNotIn("edge_attr", PoolingOutput.__dataclass_fields__)

    def test_conv_state_uses_only_explicit_role_names(self):
        model = GraphClassifier(
            2,
            2,
            _case().model,
            pool_method="nopool",
            ratio=0.5,
            avg_node_num=3,
        )
        current_state = model.state_dict()
        self.assertTrue(any(key.startswith("pre_conv.") for key in current_state))
        self.assertTrue(any(key.startswith("post_conv.") for key in current_state))
        for old_name in ("conv1.", "norm1.", "conv2.", "norm2."):
            self.assertFalse(any(key.startswith(old_name) for key in current_state))

        old_names = {
            "pre_conv.": "conv1.",
            "pre_norm.": "norm1.",
            "post_conv.": "conv2.",
            "post_norm.": "norm2.",
        }
        old_state = {}
        for key, value in current_state.items():
            old_key = key
            for current_name, old_name in old_names.items():
                if key.startswith(current_name):
                    old_key = old_name + key[len(current_name):]
                    break
            old_state[old_key] = value
        with self.assertRaises(RuntimeError):
            model.load_state_dict(old_state)

    def test_classifier_variant_only_controls_readout_merge(self):
        graph = Data(
            x=torch.randn(4, 2),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
            ),
            batch=torch.zeros(4, dtype=torch.long),
        )
        for variant, expected_readouts in (("plain", 1), ("sum", 2)):
            with self.subTest(variant=variant):
                model = GraphClassifier(
                    2,
                    2,
                    _case(variant=variant).model,
                    pool_method="nopool",
                    ratio=0.5,
                    avg_node_num=4,
                )
                with patch(
                    "gplab.model.classifier.readout",
                    wraps=graph_readout,
                ) as readout_mock:
                    model(graph)
                self.assertEqual(readout_mock.call_count, expected_readouts)

    def test_checkpoint_is_one_top_level_forward_branch(self):
        graph = Data(
            x=torch.randn(4, 2),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
            ),
            batch=torch.zeros(4, dtype=torch.long),
        )

        regular_model = GraphClassifier(
            2,
            2,
            _case().model,
            pool_method="nopool",
            ratio=0.5,
            avg_node_num=4,
        )
        with patch("gplab.model.classifier.checkpoint") as checkpoint_mock:
            regular_model(graph)
        checkpoint_mock.assert_not_called()

        checkpointed_model = GraphClassifier(
            2,
            2,
            _case().model,
            pool_method="nopool",
            ratio=0.5,
            avg_node_num=4,
            activation_checkpoint=True,
        )
        with patch(
            "gplab.model.classifier.checkpoint",
            wraps=torch_checkpoint,
        ) as checkpoint_mock:
            logits, auxiliary_loss = checkpointed_model(graph)
            self.assertIsNone(auxiliary_loss)
            (-logits[:, 0].mean()).backward()
        checkpoint_mock.assert_called_once()
        self.assertTrue(
            any(parameter.grad is not None for parameter in checkpointed_model.parameters())
        )

    def test_top_level_checkpoint_preserves_pool_auxiliary_loss(self):
        model = GraphClassifier(
            2,
            2,
            _case("diffpool").model,
            pool_method="diffpool",
            ratio=0.5,
            avg_node_num=4,
            activation_checkpoint=True,
        )
        graph = Data(
            x=torch.randn(4, 2),
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
            ),
            batch=torch.zeros(4, dtype=torch.long),
        )

        logits, auxiliary_loss = model(graph)
        self.assertIsNotNone(auxiliary_loss)
        (-logits[:, 0].mean() + auxiliary_loss).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_dense_adapter_preserves_scalar_values_in_sparse_form(self):
        adapter = DensePoolAdapter(torch.nn.Linear(2, 2), "densepool")
        output = adapter(
            torch.randn(2, 2), torch.tensor([[0, 1], [1, 0]]), torch.zeros(2, dtype=torch.long),
        )
        self.assertEqual(output.edge_weight.numel(), output.edge_index.size(1))

    def test_replay_uses_recorded_splits(self):
        case = _case()
        record = {
            "case": case.to_mapping(), "execution": ExecutionOptions(None, None, False).to_mapping(),
            "run_plan": {"case_id": "source", "seeds": [1],
                         "splits": [{"train": [2, 3], "val": [1], "test": [0]}]},
        }
        request = BenchmarkRequest.from_record_for_replay(record)
        dataset = _Dataset([
            Data(x=torch.ones(2, 2), edge_index=torch.tensor([[0], [1]])) for _ in range(4)
        ])
        with patch("gplab.experiment.execute.load_dataset", return_value=dataset):
            prepared = prepare_run(request, torch.device("cpu"), {})
        self.assertEqual(prepared.run_plan.splits[0].train, (2, 3))

    def test_comparison_cases_reuse_concrete_splits(self):
        topk_plan = RunPlan.build(_case("topkpool"), dataset_size=12)
        sag_plan = RunPlan.build(_case("sagpool"), dataset_size=12)
        self.assertEqual(topk_plan.splits, sag_plan.splits)


if __name__ == "__main__":
    unittest.main()
