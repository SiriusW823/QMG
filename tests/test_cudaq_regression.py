"""Regression tests for CUDA-Q dynamic circuit implementation.

These tests verify that the CUDA-Q implementation produces counts that match
the expected baseline. The baselines were established to ensure correctness
across different parameter configurations.
"""

import importlib.util
import json
import os
import random
import unittest

import cudaq
import numpy as np

# Import the shared reorder_counts utility
from qmg.utils.counts import reorder_counts


def _load_dynamic_builder():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    module_path = os.path.join(repo_root, "qmg", "utils", "build_dynamic_circuit.py")
    spec = importlib.util.spec_from_file_location("build_dynamic_circuit", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DynamicCircuitBuilder


DynamicCircuitBuilder = _load_dynamic_builder()


class DynamicCountsRegressionTest(unittest.TestCase):
    """Test suite for CUDA-Q dynamic circuit counts regression."""

    def _run_baseline_test(self, baseline_filename: str):
        """Helper to run a baseline comparison test.
        
        Args:
            baseline_filename: Name of the baseline JSON file in tests/data/
        """
        baseline_path = os.path.join(
            os.path.dirname(__file__), "data", baseline_filename
        )
        
        if not os.path.exists(baseline_path):
            self.skipTest(f"Baseline file {baseline_filename} not found")
        
        with open(baseline_path) as f:
            baseline = json.load(f)

        # Skip if baseline counts are not populated (placeholder file)
        expected_counts = baseline.get("counts", {})
        if not expected_counts or "__NOTE__" in expected_counts:
            self.skipTest(f"Baseline {baseline_filename} has no valid counts - needs generation")

        params = baseline["params"]
        random.seed(params["random_seed"])
        np.random.seed(params["random_seed"])
        cudaq.set_random_seed(params["random_seed"])

        builder = DynamicCircuitBuilder(
            params["num_heavy_atom"],
            temperature=params["temperature"],
            remove_bond_disconnection=params["remove_bond_disconnection"],
            chemistry_constraint=params["chemistry_constraint"],
        )
        kernel = builder.generate_quantum_circuit(
            random_seed=params["random_seed"]
        )

        # Validate that builder provides required attributes explicitly
        # (not using getattr with defaults - must fail if missing)
        self.assertTrue(
            hasattr(builder, "main_register"),
            "Builder must provide main_register attribute"
        )
        self.assertTrue(
            hasattr(builder, "main_measurement_order"),
            "Builder must provide main_measurement_order attribute"
        )
        
        # Validate main_measurement_order structure
        main_order = builder.main_measurement_order
        self.assertIsInstance(
            main_order, list,
            "main_measurement_order must be a list"
        )
        # Verify all elements are integers
        for i, elem in enumerate(main_order):
            self.assertIsInstance(
                elem, int,
                f"main_measurement_order[{i}] must be an integer"
            )

        sample_result = cudaq.sample(
            kernel, shots_count=params["num_sample"], explicit_measurements=False
        )
        counts = dict(sample_result.get_register_counts(builder.main_register).items())
        counts = reorder_counts(counts, main_order)
        
        self.assertEqual(counts, expected_counts)

    def test_dynamic_counts_baseline_seed_0(self):
        """Test dynamic circuit counts match baseline with random_seed=0, num_heavy_atom=3."""
        self._run_baseline_test("dynamic_counts_baseline.json")

    def test_dynamic_counts_baseline_seed_42(self):
        """Test dynamic circuit counts match baseline with random_seed=42, num_heavy_atom=3.
        
        This test uses a different random seed to verify bit ordering and
        conditional logic work correctly across multiple configurations.
        """
        self._run_baseline_test("dynamic_counts_baseline_seed42.json")

    def test_builder_attributes_exist_and_valid(self):
        """Test that builder always provides required attributes with correct types.
        
        This test verifies that:
        1. main_register and main_measurement_order exist on the builder
        2. main_measurement_order is a list of integers
        3. main_measurement_order length equals num_clbits after circuit generation
        """
        for num_atoms in [2, 3, 4]:
            builder = DynamicCircuitBuilder(
                num_atoms,
                temperature=0.2,
                remove_bond_disconnection=True,
                chemistry_constraint=True,
            )
            
            # Before generating circuit - attributes should exist
            self.assertTrue(
                hasattr(builder, "main_register"),
                f"Builder for {num_atoms} atoms must have main_register"
            )
            self.assertTrue(
                hasattr(builder, "main_measurement_order"),
                f"Builder for {num_atoms} atoms must have main_measurement_order"
            )

            # After generating circuit
            random.seed(0)
            np.random.seed(0)
            cudaq.set_random_seed(0)
            builder.generate_quantum_circuit(random_seed=0)
            
            main_order = builder.main_measurement_order
            self.assertIsInstance(
                main_order, list,
                f"main_measurement_order must be a list for {num_atoms} atoms"
            )
            self.assertEqual(
                len(main_order),
                builder.num_clbits,
                f"main_measurement_order length must match num_clbits for {num_atoms} atoms"
            )
            # Verify all elements are integers
            for i, elem in enumerate(main_order):
                self.assertIsInstance(
                    elem, int,
                    f"main_measurement_order[{i}] must be int for {num_atoms} atoms"
                )

    def test_no_quakevalue_arithmetic_in_conditions(self):
        """Test that no QuakeValue arithmetic is used for condition combining.
        
        This test verifies that the _or_measurements method (which used invalid
        QuakeValue arithmetic) has been removed from the DynamicCircuitBuilder.
        """
        builder = DynamicCircuitBuilder(
            num_heavy_atom=3,
            temperature=0.2,
            remove_bond_disconnection=True,
            chemistry_constraint=True,
        )
        
        # Verify _or_measurements method does not exist
        self.assertFalse(
            hasattr(builder, "_or_measurements"),
            "DynamicCircuitBuilder should not have _or_measurements method"
        )


if __name__ == "__main__":
    unittest.main()
