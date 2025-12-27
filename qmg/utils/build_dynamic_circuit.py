import cudaq
import numpy as np
import random
from typing import List, Union


class DynamicCircuitBuilder:
    """Dynamic circuit version of the molecule generation ansatz using CUDA-Q kernels."""

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
        self.num_qubits = 4 + (num_heavy_atom - 1) * 2
        self.num_clbits = num_heavy_atom * (num_heavy_atom + 1)
        self.length_all_weight_vector = int(
            8 + (self.num_heavy_atom - 2) * (self.num_heavy_atom + 3) * 3 / 2
        )
        self.main_register = "c"
        self._aux_register = "_all"
        self.main_measurement_order: List[int] = []
        self.classical_measurements = {}

    def initialize_quantum_circuit(self):
        self.kernel = cudaq.make_kernel()
        self.qubits = self.kernel.qalloc(self.num_qubits)
        self.main_measurement_order = []
        self.classical_measurements = {}

    def softmax_temperature(self, weight_vector):
        weight_vector /= self.temperature
        exps = np.exp(weight_vector)
        return exps / np.sum(exps)

    def controlled_ry(self, control: int, target: int, digit: float):
        self.kernel.cry(np.pi * digit, self.qubits[control], self.qubits[target])

    def _measure_qubits(
        self,
        qubit_indices: List[int],
        clbit_indices: List[int],
        register_name: str = None,
    ):
        if register_name is None:
            register_name = self.main_register
        handles = []
        for qubit_index, clbit_index in sorted(
            zip(qubit_indices, clbit_indices), key=lambda x: x[1]
        ):
            handle = self.kernel.mz(self.qubits[qubit_index], register_name)
            if register_name == self.main_register:
                self.classical_measurements[clbit_index] = handle
                self.main_measurement_order.append(clbit_index)
            handles.append(handle)
        return handles

    def _or_measurements(self, handles: List[cudaq.QuakeValue]):
        assert handles, "At least one measurement handle is required."
        accumulator = handles[0]
        for handle in handles[1:]:
            accumulator = (accumulator + handle) + (accumulator * handle)
        return accumulator

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

        # measure atom 1 state:
        self._measure_qubits([0, 1], [0, 1])
        # measure atom 2 state and save:
        atom_two_handles = self._measure_qubits([2, 3], [2, 3])

        atom2_condition = self._or_measurements(atom_two_handles)

        def atom_two_branch():
            self.kernel.ry(np.pi * weight_vector[6], self.qubits[4])
            self.kernel.x(self.qubits[5])
            self.kernel.cx(self.qubits[4], self.qubits[5])
            self.controlled_ry(4, 5, weight_vector[7])

        self.kernel.c_if(atom2_condition, atom_two_branch)

        self._measure_qubits([4, 5], [4, 5])

    def reset_previous_atom_bond_circuit(self, heavy_idx):
        reset_qubits_index = list(range(2, 2 * heavy_idx))
        start_clbit = (heavy_idx - 2) ** 2 + (heavy_idx - 2)
        reset_clbits_index = list(range(start_clbit, start_clbit + (heavy_idx - 1) * 2))
        for qubit_index, clbit_index in zip(reset_qubits_index, reset_clbits_index):
            handle = self.classical_measurements.get(clbit_index)
            if handle is None:
                continue
            self.kernel.c_if(
                handle, lambda idx=qubit_index: self.kernel.x(self.qubits[idx])
            )

    def _atom_existence_handles(self, heavy_atom_number: int):
        start_idx = (heavy_atom_number - 1) ** 2 + (heavy_atom_number - 1)
        return [
            self.classical_measurements[start_idx],
            self.classical_measurements[start_idx + 1],
        ]

    def build_atom_type_circuit(
        self, heavy_atom_number: int, weight_vector: Union[List[float], np.ndarray]
    ):
        assert len(weight_vector) == 3
        qubit_1_index = 2
        qubit_2_index = 3
        clbit_1_index = (heavy_atom_number - 1) ** 2 + (heavy_atom_number - 1)
        clbit_2_index = clbit_1_index + 1

        existence_handles = self._atom_existence_handles(heavy_atom_number - 1)
        existence_condition = self._or_measurements(existence_handles)

        def atom_branch():
            self.kernel.ry(np.pi * weight_vector[0], self.qubits[qubit_1_index])
            self.kernel.ry(np.pi * weight_vector[1], self.qubits[qubit_2_index])
            self.controlled_ry(qubit_1_index, qubit_2_index, weight_vector[2])

        self.kernel.c_if(existence_condition, atom_branch)
        self._measure_qubits(
            [qubit_1_index, qubit_2_index], [clbit_1_index, clbit_2_index]
        )

    def build_bond_type_circuit(
        self,
        heavy_atom_number: int,
        fixed_weight_vector: Union[List[float], np.ndarray],
        flexible_weight_vector: Union[List[float], np.ndarray],
    ):
        assert len(fixed_weight_vector) == heavy_atom_number - 1
        assert len(flexible_weight_vector) == 2 * (heavy_atom_number - 1)
        qubit_start_index = 4
        qubit_end_index = qubit_start_index + 2 * (heavy_atom_number - 1)
        clbit_start_index = heavy_atom_number ** 2 - heavy_atom_number + 2
        clbit_end_index = clbit_start_index + 2 * (heavy_atom_number - 1)

        existence_handles = self._atom_existence_handles(heavy_atom_number)
        existence_condition = self._or_measurements(existence_handles)

        def bond_branch():
            for i in range(heavy_atom_number - 1):
                self.kernel.ry(
                    np.pi * fixed_weight_vector[i], self.qubits[qubit_start_index + 2 * i + 1]
                )
                self.controlled_ry(
                    qubit_start_index + 2 * i + 1,
                    qubit_start_index + 2 * i,
                    flexible_weight_vector[2 * i],
                )
                self.controlled_ry(
                    qubit_start_index + 2 * i,
                    qubit_start_index + 2 * i + 1,
                    flexible_weight_vector[2 * i + 1],
                )

        self.kernel.c_if(existence_condition, bond_branch)

        # Measure to auxiliary register for conditional logic, then record main outputs.
        cond_handles = self._measure_qubits(
            list(range(qubit_start_index, qubit_end_index)),
            list(range(clbit_start_index, clbit_end_index)),
            register_name="_cond",
        )
        bond_condition = self._or_measurements(cond_handles)

        target_index = qubit_end_index - 1
        # Apply X when no bond is present by toggling and undoing when condition is met.
        self.kernel.x(self.qubits[target_index])
        self.kernel.c_if(
            bond_condition, lambda idx=target_index: self.kernel.x(self.qubits[idx])
        )

        self._measure_qubits(
            list(range(qubit_start_index, qubit_end_index)),
            list(range(clbit_start_index, clbit_end_index)),
        )

    def _measure_all_qubits(self):
        for qubit_index in range(self.num_qubits):
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
        used_part = 8
        for heavy_idx in range(3, self.num_heavy_atom + 1):
            num_fixed = heavy_idx - 1
            num_flexible = 2 * num_fixed
            atom_type_weight_vector = self.all_weight_vector[used_part : used_part + 3]
            bond_type_fixed_part = self.all_weight_vector[
                used_part + 3 : used_part + 3 + num_fixed
            ]
            bond_type_flexible_part = self.all_weight_vector[
                used_part + 3 + num_fixed : used_part + 3 + num_fixed + num_flexible
            ]
            used_part += 3 + num_fixed + num_flexible

            self.reset_previous_atom_bond_circuit(heavy_idx)
            self.build_atom_type_circuit(heavy_idx, atom_type_weight_vector)
            self.build_bond_type_circuit(
                heavy_idx, bond_type_fixed_part, bond_type_flexible_part
            )

        self._measure_all_qubits()
        return self.kernel


if __name__ == "__main__":
    qc_generator = DynamicCircuitBuilder(num_heavy_atom=5)
    dqc = qc_generator.generate_quantum_circuit()
    print(dqc)

