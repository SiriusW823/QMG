import cudaq
import numpy as np
import random
from typing import List, Union


class CircuitBuilder:
    """This normal circuit does not support the function of conditional weight or molecular structure generation."""

    def __init__(
        self,
        num_heavy_atom: int,
        temperature: float = 0.2,
        remove_bond_disconnection: bool = True,
        chemistry_constraint: bool = True,
    ):
        self.num_heavy_atom = num_heavy_atom
        self.temperature = temperature
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint = chemistry_constraint
        self.num_qubits = num_heavy_atom * (num_heavy_atom + 1)
        self.num_ancilla_qubits = num_heavy_atom - 1
        self.length_all_weight_vector = int(
            8 + (self.num_heavy_atom - 2) * (self.num_heavy_atom + 3) * 3 / 2
        )
        self.main_register = "c"
        self._aux_register = "_all"
        self.main_measurement_order: List[int] = []

    def initialize_quantum_circuit(self):
        self.kernel = cudaq.make_kernel()
        self.qubits = self.kernel.qalloc(self.num_qubits + self.num_ancilla_qubits)
        self.main_measurement_order = []

    def softmax_temperature(self, weight_vector):
        weight_vector /= self.temperature
        exps = np.exp(weight_vector)
        return exps / np.sum(exps)

    def controlled_ry(self, control: int, target: int, digit: float):
        self.kernel.cry(np.pi * digit, self.qubits[control], self.qubits[target])

    def _or_measurements(self, handles: List[cudaq.QuakeValue]):
        assert handles, "At least one measurement handle is required."
        accumulator = handles[0]
        for handle in handles[1:]:
            accumulator = (accumulator + handle) + (accumulator * handle)
        return accumulator

    def ccx(self, control_1: int, control_2: int, target: int):
        base, ctrl, tgt = cudaq.make_kernel(cudaq.qubit, cudaq.qubit)
        base.cx(ctrl, tgt)
        self.kernel.control(
            base,
            self.qubits[control_1],
            self.qubits[control_2],
            self.qubits[target],
        )

    def build_two_atoms(self, weight_vector: Union[List[float], np.ndarray]):
        assert len(weight_vector) == 8  # length of weight vector should be 8
        self.kernel.ry(np.pi * weight_vector[0], self.qubits[0])
        self.kernel.x(self.qubits[1])
        self.kernel.ry(np.pi * weight_vector[2], self.qubits[2])
        self.kernel.ry(np.pi * weight_vector[4], self.qubits[3])
        self.kernel.cx(self.qubits[0], self.qubits[1])
        self.controlled_ry(1, 2, weight_vector[3])
        self.kernel.cx(self.qubits[2], self.qubits[3])
        self.controlled_ry(0, 1, weight_vector[1])
        self.kernel.cx(self.qubits[1], self.qubits[2])
        self.controlled_ry(2, 3, weight_vector[5])
        self.kernel.x(self.qubits[2])
        self.kernel.x(self.qubits[3])
        self.kernel.x(self.qubits[4])
        self.ccx(2, 3, 4)
        self.kernel.x(self.qubits[2])
        self.kernel.x(self.qubits[3])
        self.kernel.cx(self.qubits[4], self.qubits[5])
        self.kernel.cx(self.qubits[5], self.qubits[6])
        self.controlled_ry(4, 5, weight_vector[6])
        self.kernel.cx(self.qubits[5], self.qubits[6])
        self.controlled_ry(5, 6, weight_vector[7])

    def build_bond_type_circuit(
        self,
        heavy_atom_number: int,
        fixed_weight_vector: Union[List[float], np.ndarray],
        flexible_weight_vector: Union[List[float], np.ndarray],
    ):
        num_target_qubit = (heavy_atom_number - 1) * 2
        assert len(fixed_weight_vector) * 2 == num_target_qubit == len(
            flexible_weight_vector
        )
        ancilla_qubit_index = 2 * heavy_atom_number + (heavy_atom_number - 1) ** 2 - 1
        for i in range(2 * (heavy_atom_number - 1)):
            self.kernel.cx(
                self.qubits[ancilla_qubit_index + i], self.qubits[ancilla_qubit_index + i + 1]
            )
        for i in range((heavy_atom_number - 1)):
            self.controlled_ry(
                ancilla_qubit_index + num_target_qubit - 2 * i - 1,
                ancilla_qubit_index + num_target_qubit - 2 * i,
                1 - fixed_weight_vector[-1 - i],
            )
            self.kernel.cx(
                self.qubits[ancilla_qubit_index + num_target_qubit - 2 * i - 2],
                self.qubits[ancilla_qubit_index + num_target_qubit - 2 * i - 1],
            )
            self.controlled_ry(
                ancilla_qubit_index + num_target_qubit - 2 * i,
                ancilla_qubit_index + num_target_qubit - 2 * i - 1,
                flexible_weight_vector[-2 - 2 * i],
            )
            self.controlled_ry(
                ancilla_qubit_index + num_target_qubit - 2 * i - 1,
                ancilla_qubit_index + num_target_qubit - 2 * i,
                flexible_weight_vector[-1 - 2 * i],
            )

    def build_atom_type_circuit(
        self, heavy_atom_number: int, weight_vector: Union[List[float], np.ndarray]
    ):
        assert len(weight_vector) == 3
        ancilla_qubit_index = 2 * (heavy_atom_number - 1) + (heavy_atom_number - 2) ** 2 - 1
        qubit_1_index = ancilla_qubit_index + 2 * (heavy_atom_number - 2) + 1
        qubit_2_index = qubit_1_index + 1
        self.kernel.cx(self.qubits[ancilla_qubit_index], self.qubits[qubit_1_index])
        self.controlled_ry(qubit_1_index, qubit_2_index, weight_vector[1])
        self.controlled_ry(ancilla_qubit_index, qubit_1_index, weight_vector[0])
        self.kernel.cx(self.qubits[qubit_2_index], self.qubits[qubit_1_index])
        self.controlled_ry(qubit_1_index, qubit_2_index, weight_vector[2])
        self.kernel.x(self.qubits[qubit_1_index])
        self.kernel.x(self.qubits[qubit_2_index])
        self.kernel.x(self.qubits[qubit_2_index + 1])
        self.ccx(qubit_1_index, qubit_2_index, qubit_2_index + 1)
        self.kernel.x(self.qubits[qubit_1_index])
        self.kernel.x(self.qubits[qubit_2_index])

    def build_removing_bond_disconnection_circuit(self, heavy_atom_number: int):
        ancilla_qubit_index = 2 * (heavy_atom_number) + (heavy_atom_number - 1) ** 2 - 1
        control_qubits_index_list = list(
            range(
                ancilla_qubit_index + 1,
                ancilla_qubit_index + 1 + 2 * (heavy_atom_number - 1),
            )
        )
        cond_handles = [
            self.kernel.mz(self.qubits[idx], "_cond") for idx in control_qubits_index_list
        ]
        bond_condition = self._or_measurements(cond_handles)
        target_qubit = control_qubits_index_list[-1]
        self.kernel.x(self.qubits[target_qubit])
        self.kernel.c_if(
            bond_condition, lambda idx=target_qubit: self.kernel.x(self.qubits[idx])
        )

    def measure(self):
        effective_qubit_index = list(range(self.num_qubits + self.num_ancilla_qubits))
        for j in range(2, self.num_heavy_atom + 1):
            ancilla_qubit_number = 2 * j + (j - 1) ** 2 - 1
            effective_qubit_index.remove(ancilla_qubit_number)
        self.effective_qubit_index = effective_qubit_index
        self.main_measurement_order = list(range(self.num_qubits))
        for qubit_index in self.effective_qubit_index:
            self.kernel.mz(self.qubits[qubit_index], self.main_register)
        for qubit_index in range(self.num_qubits + self.num_ancilla_qubits):
            self.kernel.mz(self.qubits[qubit_index], self._aux_register)

    def generate_quantum_circuit(
        self, all_weight_vector: Union[List[float], np.ndarray] = None, random_seed: int = 0
    ):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.initialize_quantum_circuit()
        # (1) generate weight vector
        if isinstance(all_weight_vector, (np.ndarray, list)):
            assert len(all_weight_vector) == self.length_all_weight_vector
            self.all_weight_vector = all_weight_vector
        else:
            self.all_weight_vector = np.array(
                [random.random() for _ in range(self.length_all_weight_vector)]
            )
            if self.chemistry_constraint and (self.num_heavy_atom >= 3):
                used_part = 8
                for heavy_idx in range(3, self.num_heavy_atom + 1):
                    used_part += 3  # atom type weight vector
                    num_fixed = heavy_idx - 1
                    num_flexible = 2 * num_fixed
                    bond_type_fixed_part = self.all_weight_vector[
                        used_part : used_part + num_fixed
                    ]
                    self.all_weight_vector[
                        used_part : used_part + num_fixed
                    ] = self.softmax_temperature(bond_type_fixed_part)
                    bond_type_flexible_part = self.all_weight_vector[
                        used_part + num_fixed : used_part + num_fixed + num_flexible
                    ]
                    bond_type_flexible_part *= 0.5
                    bond_type_flexible_part += np.array([0, 0.5] * (heavy_idx - 1))
                    self.all_weight_vector[
                        used_part + num_fixed : used_part + num_fixed + num_flexible
                    ] = bond_type_flexible_part
                    used_part += num_fixed + num_flexible
        # (2) start to construct the quantum circuit
        self.build_two_atoms(self.all_weight_vector[0:8])
        if self.num_heavy_atom - 2 > 0:
            used_part = 8
            for heavy_idx in range(3, self.num_heavy_atom + 1):
                num_fixed = heavy_idx - 1
                num_flexible = 2 * num_fixed
                atom_type_weight_vector = self.all_weight_vector[
                    used_part : used_part + 3
                ]
                bond_type_fixed_part = self.all_weight_vector[
                    used_part + 3 : used_part + 3 + num_fixed
                ]
                bond_type_flexible_part = self.all_weight_vector[
                    used_part + 3 + num_fixed : used_part + 3 + num_fixed + num_flexible
                ]
                used_part += 3 + num_fixed + num_flexible
                self.build_atom_type_circuit(heavy_idx, atom_type_weight_vector)
                if (heavy_idx >= 4) and self.remove_bond_disconnection:
                    self.build_removing_bond_disconnection_circuit(heavy_idx - 1)
                self.build_bond_type_circuit(
                    heavy_idx, bond_type_fixed_part, bond_type_flexible_part
                )
            else:
                if self.remove_bond_disconnection:
                    self.build_removing_bond_disconnection_circuit(heavy_idx)
        self.measure()
        return self.kernel


if __name__ == "__main__":
    qc_generator = CircuitBuilder(num_heavy_atom=3)
    qc = qc_generator.generate_quantum_circuit()
    print(qc)

