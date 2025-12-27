import cudaq
import numpy as np
import random
from typing import List, Union

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from .utils import (
    MoleculeQuantumStateGenerator,
    CircuitBuilder,
    DynamicCircuitBuilder,
    ConditionalWeightsGenerator,
)


class MoleculeGenerator:
    def __init__(
        self,
        num_heavy_atom: int,
        all_weight_vector: Union[List[float], np.ndarray] = None,
        backend_name: str = "cudaq",
        temperature: float = 0.2,
        dynamic_circuit: bool = True,
        remove_bond_disconnection: bool = True,
        chemistry_constraint: bool = True,
    ):
        self.num_heavy_atom = num_heavy_atom
        self.all_weight_vector = all_weight_vector
        self.backend_name = backend_name
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint = chemistry_constraint
        self.temperature = temperature
        self.dynamic_circuit = dynamic_circuit
        self.num_qubits = num_heavy_atom * (num_heavy_atom + 1)
        self.num_ancilla_qubits = num_heavy_atom - 1
        self.main_register = "c"
        self.measurement_order = []
        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size=num_heavy_atom, ncpus=1, sanitize_method="strict"
        )

    def _reorder_counts(self, counts: dict):
        if not self.measurement_order:
            return counts
        if self.measurement_order == list(range(len(self.measurement_order))):
            return counts
        reordered = {}
        for bitstring, value in counts.items():
            bits = list(str(bitstring))
            if len(bits) != len(self.measurement_order):
                reordered[str(bitstring)] = reordered.get(str(bitstring), 0) + value
                continue
            arranged = ["0"] * len(self.measurement_order)
            for position, classical_index in enumerate(self.measurement_order):
                arranged[classical_index] = bits[position]
            key = "".join(arranged)
            reordered[key] = reordered.get(key, 0) + value
        return reordered

    def generate_quantum_circuit(self, random_seed):
        if self.dynamic_circuit:
            builder = DynamicCircuitBuilder(
                self.num_heavy_atom,
                self.temperature,
                self.remove_bond_disconnection,
                self.chemistry_constraint,
            )
        else:
            builder = CircuitBuilder(
                self.num_heavy_atom,
                self.temperature,
                self.remove_bond_disconnection,
                self.chemistry_constraint,
            )
        self.qc = builder.generate_quantum_circuit(
            self.all_weight_vector, random_seed
        )
        self.main_register = getattr(builder, "main_register", "c")
        self.measurement_order = getattr(builder, "main_measurement_order", [])

    def update_weight_vector(self, all_weight_vector):
        self.all_weight_vector = all_weight_vector

    def sample_molecule(self, num_sample, random_seed: int = 0):
        random.seed(random_seed)
        np.random.seed(random_seed)
        cudaq.set_random_seed(random_seed)
        self.generate_quantum_circuit(random_seed)
        sample_result = cudaq.sample(
            self.qc, shots_count=num_sample, explicit_measurements=False
        )
        raw_counts = sample_result.get_register_counts(self.main_register)
        counts = self._reorder_counts(dict(raw_counts.items()))

        smiles_dict = {}
        num_valid_molecule = 0
        for key, value in counts.items():
            if self.dynamic_circuit:
                key = "".join(str(key).split())
            smiles = self.data_generator.QuantumStateToSmiles(
                self.data_generator.post_process_quantum_state(key)
            )
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + value
            if smiles:
                num_valid_molecule += value
        validity = num_valid_molecule / num_sample
        uniqueness = (len(smiles_dict.keys()) - 1) / num_valid_molecule
        return smiles_dict, validity, uniqueness


if __name__ == "__main__":
    num_heavy_atom = 5
    random_seed = 3
    cwg = ConditionalWeightsGenerator(
        num_heavy_atom, smarts="[O:1]1[C:2][C:3]1", disable_connectivity_position=[1]
    )
    random_weight_vector = cwg.generate_conditional_random_weights(random_seed)
    mg = MoleculeGenerator(
        num_heavy_atom, all_weight_vector=random_weight_vector, dynamic_circuit=True
    )
    smiles_dict, validity, diversity = mg.sample_molecule(20000)
    print(smiles_dict)
    print("Validity: {:.2f}%".format(validity * 100))
    print("Diversity: {:.2f}%".format(diversity * 100))
