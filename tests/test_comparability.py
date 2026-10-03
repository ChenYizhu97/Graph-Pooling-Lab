"""Pool-set verdicts and input-conditioned signature relations."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from gplab.benchmark.case import ModelConfig
from gplab.benchmark.comparability import ComparisonSetting, check_comparability
from gplab.benchmark.compatibility import pool_compatibility_error, validate_pool_compatibility
from gplab.data.profiles import DatasetProfile
from gplab.graph import ConnectivityType
from gplab.layers.conv.profiles import CONV_PROFILES, ConvProfile
from gplab.layers.pool.profiles import POOLING_PROFILES, PoolingProfile, PoolingSignature

U, W = ConnectivityType.BINARY, ConnectivityType.SCALAR


def _profile(*pairs):
    return PoolingProfile(lambda *args: None, {PoolingSignature(*pair) for pair in pairs})


class SignatureCompatibilityTests(unittest.TestCase):
    def test_builtin_domains_match_alignment_audit(self):
        expected = {
            "nopool": {(U, U), (W, W)},
            "topkpool": {(U, U)}, "sagpool": {(U, U)}, "sparsepool": {(U, U)},
            "asapool": {(U, W)}, "diffpool": {(U, W)},
            "mincutpool": {(U, W)}, "densepool": {(U, W)},
        }
        for name, pairs in expected.items():
            with self.subTest(pool=name):
                self.assertEqual(POOLING_PROFILES[name].signatures,
                                 frozenset(PoolingSignature(*pair) for pair in pairs))

    def test_signature_collections_are_immutable_and_deduplicated(self):
        signature = PoolingSignature(U, U)
        for collection in ([signature, signature], (signature,), {signature}, frozenset({signature})):
            with self.subTest(collection=type(collection).__name__):
                profile = PoolingProfile(lambda *args: None, collection)
                self.assertIsInstance(profile.signatures, frozenset)
                self.assertEqual(profile.signatures, frozenset({signature}))
        with self.assertRaises(ValueError):
            PoolingProfile(lambda *args: None, set())
        with self.assertRaises(TypeError):
            PoolingProfile(lambda *args: None, {(U, U)})
        with self.assertRaises(TypeError):
            PoolingSignature("binary", W)

    def test_input_membership_and_conditional_outputs(self):
        profile = _profile((U, U), (W, W))
        self.assertEqual(profile.input_types, {U, W})
        self.assertEqual(profile.output_types_for(U), {U})
        self.assertEqual(profile.output_types_for(W), {W})
        with patch("gplab.benchmark.compatibility.load_pooling_profile", return_value=profile):
            # The W -> W alternative must not force a U dataset to use a weighted post-conv.
            for input_type, post_conv, allowed in ((U, "GIN", True), (W, "GIN", False), (W, "GCN", True)):
                with self.subTest(input_type=input_type, post_conv=post_conv):
                    error = pool_compatibility_error(dataset_type=input_type, pool_name="custom",
                                                     pre_conv="GIN", post_conv=post_conv)
                    self.assertEqual(error is None, allowed)

    def test_every_possible_output_must_be_consumable(self):
        profile = _profile((U, U), (U, W), (W, W))
        self.assertEqual(profile.output_types_for(U), {U, W})
        with patch("gplab.benchmark.compatibility.load_pooling_profile", return_value=profile):
            validate_pool_compatibility(dataset_type=U, pool_name="custom", pre_conv="GCN", post_conv="GCN")
            with self.assertRaisesRegex(ValueError, "cannot consume scalar"):
                validate_pool_compatibility(dataset_type=U, pool_name="custom", pre_conv="GCN", post_conv="GIN")
            # Overlap is insufficient in either direction: a scalar-only encoder
            # cannot safely process the binary alternative either.
            scalar_only = ConvProfile(lambda *args: None, frozenset({W}))
            with patch("gplab.benchmark.compatibility.CONV_PROFILES", {**CONV_PROFILES, "scalar_only": scalar_only}):
                with self.assertRaisesRegex(ValueError, "cannot consume binary"):
                    validate_pool_compatibility(dataset_type=U, pool_name="custom", pre_conv="GCN", post_conv="scalar_only")

    def test_raw_input_values_cannot_bypass_signature_matching(self):
        # str-enum equality must not let a string pass input membership while
        # failing the enum-identity filter used to determine possible outputs.
        for value in ("binary", None, 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                validate_pool_compatibility(dataset_type=value, pool_name="diffpool", pre_conv="GCN", post_conv="GIN")

    def test_unsupported_input_does_not_pass_via_an_empty_output_set(self):
        profile = _profile((U, W))
        self.assertEqual(profile.output_types_for(W), frozenset())
        with patch("gplab.benchmark.compatibility.load_pooling_profile", return_value=profile):
            with self.assertRaisesRegex(ValueError, "not declared valid"):
                validate_pool_compatibility(dataset_type=W, pool_name="custom", pre_conv="GCN", post_conv="GCN")


class ComparabilityTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelConfig(hidden_features=4, nonlinearity="relu", p_dropout=0.0,
                                 pre_conv="GCN", post_conv="GCN", pre_gnn=(4,), post_gnn=(8, 4), variant="plain")
        self.setting = ComparisonSetting(dataset="MUTAG", model=self.model)

    def test_verdict_describes_pools_and_setting_not_one_signature(self):
        result = check_comparability(["topkpool", "diffpool"], self.setting)
        self.assertTrue(result.comparable)  # U and W outputs need not match each other.
        self.assertEqual(result.pools, ("topkpool", "diffpool"))
        self.assertEqual(result.setting, self.setting)
        self.assertEqual(result.incompatibilities, {})

    def test_incompatible_setting_reports_each_rejected_pool(self):
        setting = replace(self.setting, model=replace(self.model, post_conv="GIN"))
        result = check_comparability(["topkpool", "diffpool", "asapool"], setting)
        self.assertFalse(result.comparable)
        self.assertEqual(set(result.incompatibilities[U]), {"diffpool", "asapool"})
        self.assertTrue(all("scalar" in reason for reason in result.incompatibilities[U].values()))

    def test_accepts_custom_profiles_and_set_input(self):
        result = check_comparability({"nopool", "examples.custom_pool_plugin:CUSTOM_POOL_PROFILE"}, self.setting)
        self.assertTrue(result.comparable)
        self.assertEqual(len(result.pools), 2)

    def test_dataset_profile_changes_the_verdict_without_loading_data(self):
        def unexpected_load():
            self.fail("Structural comparability must not load datasets")
        with patch("gplab.benchmark.comparability.get_dataset_profile", return_value=DatasetProfile(unexpected_load, W)):
            result = check_comparability(["nopool", "topkpool", "diffpool"], self.setting)
        self.assertTrue(result.comparable)
        self.assertEqual(result.input_types, {U})
        self.assertEqual(set(result.incompatibilities[W]), {"topkpool", "diffpool"})

    def test_shared_input_must_exist_not_just_individual_compatibility(self):
        profiles = {"binary_only": _profile((U, U)), "scalar_only": _profile((W, W))}
        with patch("gplab.benchmark.comparability.get_dataset_profile", return_value=DatasetProfile(lambda: None, W)), \
                patch("gplab.benchmark.compatibility.load_pooling_profile", side_effect=profiles.__getitem__):
            result = check_comparability(profiles, self.setting)
        self.assertFalse(result.comparable)
        self.assertEqual(result.input_types, frozenset())
        self.assertEqual(set(result.incompatibilities[U]), {"scalar_only"})
        self.assertEqual(set(result.incompatibilities[W]), {"binary_only"})

    def test_outputs_filter_shared_inputs_and_explicit_choice_cannot_fallback(self):
        profiles = {"a": _profile((U, U), (W, W)), "b": _profile((U, U), (W, W))}
        with patch("gplab.benchmark.comparability.get_dataset_profile", return_value=DatasetProfile(lambda: None, W)), \
                patch("gplab.benchmark.compatibility.load_pooling_profile", side_effect=profiles.__getitem__):
            self.assertEqual(check_comparability(profiles, self.setting).input_types, {U, W})
            setting = replace(self.setting, model=replace(self.model, post_conv="GIN"))
            self.assertEqual(check_comparability(profiles, setting).input_types, {U})
            self.assertFalse(check_comparability(profiles, replace(setting, input_type=W)).comparable)

    def test_binary_dataset_cannot_provide_scalar_input(self):
        profiles = {"a": _profile((W, W)), "b": _profile((W, W))}
        with patch("gplab.benchmark.compatibility.load_pooling_profile", side_effect=profiles.__getitem__):
            self.assertFalse(check_comparability(profiles, self.setting).comparable)
        with self.assertRaisesRegex(ValueError, "cannot provide scalar"):
            check_comparability(["nopool", "topkpool"], replace(self.setting, input_type=W))
        with self.assertRaises(TypeError):
            replace(self.setting, input_type="binary")

    def test_comparison_requires_two_distinct_profiles(self):
        for names in ([], ["nopool"], ["nopool", "nopool"]):
            with self.subTest(pools=names), self.assertRaises(ValueError):
                check_comparability(names, self.setting)
        with self.assertRaises(TypeError):
            check_comparability("nopool", self.setting)
        result = check_comparability(["nopool", "topkpool", "nopool"], self.setting)
        self.assertEqual(result.pools, ("nopool", "topkpool"))


class DatasetRepresentationTests(unittest.TestCase):
    def test_binary_projection_preserves_source_topology_and_split_batches(self):
        graph = Data(x=torch.ones(2, 1), edge_index=torch.tensor([[0, 1], [1, 0]]),
                     edge_weight=torch.tensor([0.25, 0.75]), y=torch.tensor([1]))
        source = InMemoryDataset()
        source.data, source.slices = source.collate([graph, graph.clone()])

        def transform(data):
            # Projection must also strip weights produced by an existing transform.
            data.edge_weight = data.edge_weight * 2
            data.x = data.x + 1
            return data

        source.transform = transform
        profile = DatasetProfile(lambda: source, W)
        native, binary = profile.build(), profile.build(U)
        self.assertEqual(profile.connectivity_types, {U, W})
        self.assertEqual(native.connectivity_type, W)
        self.assertEqual(binary.connectivity_type, U)
        self.assertIs(source.transform, transform)
        self.assertTrue(torch.equal(native[0].edge_weight, torch.tensor([0.5, 1.5])))
        self.assertIsNone(binary[0].edge_weight)
        self.assertTrue(torch.equal(binary[0].edge_index, graph.edge_index))
        self.assertTrue(torch.equal(binary[0].x, graph.x + 1))
        self.assertTrue(torch.equal(binary[0].y, graph.y))
        batch = next(iter(DataLoader(binary[[1, 0]], batch_size=2)))
        self.assertIsNone(batch.edge_weight)
        self.assertEqual(batch.num_edges, 4)
        self.assertTrue(torch.equal(source.get(0).edge_weight, graph.edge_weight))

    def test_binary_profile_does_not_invent_scalar_values(self):
        def unexpected_load():
            self.fail("Unsupported representations must fail before loading data")
        profile = DatasetProfile(unexpected_load, U)
        self.assertEqual(profile.connectivity_types, {U})
        with self.assertRaisesRegex(ValueError, "cannot provide scalar"):
            profile.build(W)
        with self.assertRaises(TypeError):
            profile.build("binary")
