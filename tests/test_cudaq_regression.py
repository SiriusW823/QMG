import importlib.util
import json
import os
import random
import unittest

import cudaq
import numpy as np


def _load_dynamic_builder():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    module_path = os.path.join(repo_root, "qmg", "utils", "build_dynamic_circuit.py")
    spec = importlib.util.spec_from_file_location("build_dynamic_circuit", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DynamicCircuitBuilder


DynamicCircuitBuilder = _load_dynamic_builder()


def _reorder_counts(counts: dict, order):
    if not order or order == list(range(len(order))):
        return counts
    reordered = {}
    for bitstring, value in counts.items():
        bits = list(str(bitstring))
        if len(bits) != len(order):
            reordered[str(bitstring)] = reordered.get(str(bitstring), 0) + value
            continue
        arranged = ["0"] * len(order)
        for position, classical_index in enumerate(order):
            arranged[classical_index] = bits[position]
        key = "".join(arranged)
        reordered[key] = reordered.get(key, 0) + value
    return reordered


class DynamicCountsRegressionTest(unittest.TestCase):
    def test_dynamic_counts_match_baseline(self):
        baseline_path = os.path.join(
            os.path.dirname(__file__), "data", "dynamic_counts_baseline.json"
        )
        with open(baseline_path) as f:
            baseline = json.load(f)

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

        sample_result = cudaq.sample(
            kernel, shots_count=params["num_sample"], explicit_measurements=False
        )
        counts = dict(sample_result.get_register_counts(builder.main_register).items())
        counts = _reorder_counts(counts, getattr(builder, "main_measurement_order", []))
        self.assertEqual(counts, baseline["counts"])


if __name__ == "__main__":
    unittest.main()
