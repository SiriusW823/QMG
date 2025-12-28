import cudaq
import numpy as np
import random
from typing import List, Union


class DynamicCircuitBuilder:
    """Dynamic circuit version of the molecule generation ansatz using CUDA-Q kernels.
    
    This class builds quantum circuits for molecule generation with mid-circuit
    measurements and conditional branches using CUDA-Q. The conditional logic
    uses the "ancilla flag" pattern to satisfy CUDA-Q's requirement that c_if()
    accepts only a single measurement handle.
    
    Measurement Strategy (Main + Aux Registers):
    -------------------------------------------
    - Main register ("c"): Contains the semantic output bits in the expected
      classical bit order. Use sample_result.get_register_counts("c") to
      retrieve the primary counts that match the original Qiskit output format.
    - Aux register ("_all"): Contains measurements of ALL qubits (including
      ancillas) to prevent CUDA-Q from eliding unused qubits. This register
      is for debugging only and does NOT affect the main output.
    - Condition register ("_cond"): Temporary register for intermediate
      condition measurements used in the toggle-untoggle pattern.
    
    Conditional Logic Implementation:
    ---------------------------------
    CUDA-Q's c_if() only accepts a single measurement handle (from a single
    qubit mz). To implement conditions like "if any of {q1, q2} is 1" (OR logic),
    we use the following pattern:
    
    1. Allocate a flag ancilla qubit (initialized to |0>)
    2. Compute the OR condition in the quantum domain:
       - Apply X to flag (flag = 1, meaning "condition met")
       - For each qubit qi in the condition set:
         Apply X(qi), then CX(qi, flag), then X(qi)
         This flips flag to 0 if qi was 0.
       - After all qubits: flag = 1 iff at least one qi was 1
    3. Measure the flag qubit to get a single measurement handle
    4. Use c_if(flag_handle, branch) for the conditional execution
    
    This is equivalent to the original Qiskit if_test((register, 0)) which
    executed the else-branch when the register was non-zero.
    """

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
        # Base qubits for the circuit (excluding flag ancillas)
        self._base_qubits = 4 + (num_heavy_atom - 1) * 2
        # We need flag ancilla qubits for OR conditions:
        # - 1 for atom 2 existence check in build_two_atoms
        # - 1 per heavy atom >= 3 for existence checks in build_atom_type_circuit
        # - 1 per heavy atom >= 3 for bond condition in build_bond_type_circuit
        # Total additional: 1 + 2*(num_heavy_atom - 2) for num_heavy_atom >= 3
        self._num_flag_ancillas = max(0, 1 + 2 * (num_heavy_atom - 2)) if num_heavy_atom >= 2 else 0
        self.num_qubits = self._base_qubits + self._num_flag_ancillas
        self.num_clbits = num_heavy_atom * (num_heavy_atom + 1)
        self.length_all_weight_vector = int(
            8 + (self.num_heavy_atom - 2) * (self.num_heavy_atom + 3) * 3 / 2
        )
        self.main_register = "c"
        self._aux_register = "_all"
        self.main_measurement_order: List[int] = []
        self.classical_measurements = {}
        self._flag_index = 0  # Counter for allocating flag ancillas

    def initialize_quantum_circuit(self):
        self.kernel = cudaq.make_kernel()
        self.qubits = self.kernel.qalloc(self.num_qubits)
        self.main_measurement_order = []
        self.classical_measurements = {}
        self._flag_index = 0

    def _alloc_flag_qubit(self) -> int:
        """Allocate the next available flag ancilla qubit index."""
        idx = self._base_qubits + self._flag_index
        self._flag_index += 1
        return idx

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
        """Measure specified qubits and record to the given register.
        
        Args:
            qubit_indices: List of qubit indices to measure.
            clbit_indices: List of classical bit indices (for ordering).
            register_name: Name of the register to record measurements.
                If None, uses the main register.
        
        Returns:
            List of measurement handles from mz operations.
        """
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

    def _compute_or_condition(self, qubit_indices: List[int]) -> int:
        """Compute OR of multiple qubits into a flag ancilla qubit.
        
        This implements: flag = q[0] OR q[1] OR ... using De Morgan's law:
        OR = NOT(AND(NOT(q[0]), NOT(q[1]), ...))
        
        The computation is done in-place on the qubits (they are restored
        to their original state after the computation).
        
        Equivalence to Qiskit if_test((register, 0)):
        - Qiskit if_test((reg, 0)) executes the if-branch when reg == 0
        - The else-branch executes when reg != 0 (i.e., any bit is 1)
        - This function computes flag = 1 when any input qubit is 1
        - Using c_if(flag_measurement, branch) is equivalent to the else-branch
        
        Args:
            qubit_indices: List of qubit indices to OR together.
        
        Returns:
            Index of the flag qubit containing the OR result.
        """
        flag_idx = self._alloc_flag_qubit()
        
        # Initialize flag to 1 (assume condition is met)
        self.kernel.x(self.qubits[flag_idx])
        
        if len(qubit_indices) == 0:
            # No qubits to check, flag stays 0 (no condition met)
            self.kernel.x(self.qubits[flag_idx])  # Undo the earlier X, flag = 0
            return flag_idx
        
        if len(qubit_indices) == 1:
            # Single qubit: flag = q0
            # flag starts at 1, so: if q0=0, flip flag to 0
            qi = qubit_indices[0]
            self.kernel.x(self.qubits[qi])
            self.kernel.cx(self.qubits[qi], self.qubits[flag_idx])
            self.kernel.x(self.qubits[qi])
            return flag_idx
        
        if len(qubit_indices) == 2:
            # Two qubits: use CCX (Toffoli)
            # flag = 1 initially
            # We want: flag = 0 if both q0=0 AND q1=0, else flag = 1
            q0, q1 = qubit_indices[0], qubit_indices[1]
            # Invert controls
            self.kernel.x(self.qubits[q0])
            self.kernel.x(self.qubits[q1])
            # CCX: flip flag if both inverted are 1 (both original 0)
            self._ccx(q0, q1, flag_idx)
            # Restore
            self.kernel.x(self.qubits[q0])
            self.kernel.x(self.qubits[q1])
            return flag_idx
        
        # For more than 2 qubits, we need a cascade or decomposition
        # Using iterative approach: compute partial AND into temporary qubits
        # For now, we'll handle up to 4 qubits with a cascade
        
        n = len(qubit_indices)
        
        # Invert all input qubits
        for qi in qubit_indices:
            self.kernel.x(self.qubits[qi])
        
        if n == 3:
            # Use one extra ancilla for 3-qubit AND
            aux_idx = self._alloc_flag_qubit()
            q0, q1, q2 = qubit_indices
            # Compute q0 AND q1 into aux
            self._ccx(q0, q1, aux_idx)
            # Compute aux AND q2 into flag
            self._ccx(aux_idx, q2, flag_idx)
            # Uncompute aux
            self._ccx(q0, q1, aux_idx)
        elif n == 4:
            # Use two extra ancillas for 4-qubit AND
            aux1_idx = self._alloc_flag_qubit()
            aux2_idx = self._alloc_flag_qubit()
            q0, q1, q2, q3 = qubit_indices
            # Compute q0 AND q1 into aux1
            self._ccx(q0, q1, aux1_idx)
            # Compute q2 AND q3 into aux2
            self._ccx(q2, q3, aux2_idx)
            # Compute aux1 AND aux2 into flag
            self._ccx(aux1_idx, aux2_idx, flag_idx)
            # Uncompute
            self._ccx(q2, q3, aux2_idx)
            self._ccx(q0, q1, aux1_idx)
        else:
            # For n > 4, we'd need more complex decomposition.
            # In the current molecule generator, the maximum OR condition is 4 qubits
            # (for bond checks with heavy_atom_number=5), so this limit is sufficient.
            # If larger molecules are needed, implement a tree decomposition approach.
            raise ValueError(f"OR condition with {n} qubits not supported (max 4)")
        
        # Restore all input qubits
        for qi in qubit_indices:
            self.kernel.x(self.qubits[qi])
        
        return flag_idx

    def _ccx(self, control1: int, control2: int, target: int):
        """Apply a Toffoli (CCX) gate using CUDA-Q's control mechanism."""
        # Create a sub-kernel for CX
        base, ctrl, tgt = cudaq.make_kernel(cudaq.qubit, cudaq.qubit)
        base.cx(ctrl, tgt)
        # Apply with two controls
        self.kernel.control(
            base,
            self.qubits[control1],
            self.qubits[control2],
            self.qubits[target],
        )

    def build_two_atoms(self, weight_vector: Union[List[float], np.ndarray]):
        """Build the circuit portion for the first two atoms.
        
        This applies rotations and entangling gates, then measures qubits 0-3
        for atom states. If atom 2 exists (any of qubits 2,3 measured as 1),
        additional gates are applied for bond information.
        """
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

        # Compute OR condition for atom 2 existence BEFORE measuring
        # This is equivalent to Qiskit's if_test((register, 0)) else-branch
        flag_idx = self._compute_or_condition([2, 3])
        
        # Measure the flag to get a single measurement handle for c_if
        flag_handle = self.kernel.mz(self.qubits[flag_idx], "_flag")
        
        # measure atom 1 state:
        self._measure_qubits([0, 1], [0, 1])
        # measure atom 2 state:
        self._measure_qubits([2, 3], [2, 3])

        # Conditional branch: execute if atom 2 exists (flag = 1)
        def atom_two_branch():
            self.kernel.ry(np.pi * weight_vector[6], self.qubits[4])
            self.kernel.x(self.qubits[5])
            self.kernel.cx(self.qubits[4], self.qubits[5])
            self.controlled_ry(4, 5, weight_vector[7])

        self.kernel.c_if(flag_handle, atom_two_branch)

        self._measure_qubits([4, 5], [4, 5])

    def reset_previous_atom_bond_circuit(self, heavy_idx):
        """Reset qubits from previous atom/bond circuits based on measurements.
        
        This conditionally applies X gates to reset qubits that were measured
        as 1 in previous rounds, preparing them for reuse in the current round.
        Each c_if uses a single measurement handle, conforming to CUDA-Q spec.
        """
        reset_qubits_index = list(range(2, 2 * heavy_idx))
        start_clbit = (heavy_idx - 2) ** 2 + (heavy_idx - 2)
        reset_clbits_index = list(range(start_clbit, start_clbit + (heavy_idx - 1) * 2))
        for qubit_index, clbit_index in zip(reset_qubits_index, reset_clbits_index):
            handle = self.classical_measurements.get(clbit_index)
            if handle is None:
                continue
            # Each c_if uses a single measurement handle (conforming to CUDA-Q spec)
            self.kernel.c_if(
                handle, lambda idx=qubit_index: self.kernel.x(self.qubits[idx])
            )

    def _atom_existence_qubit_indices(self, heavy_atom_number: int) -> List[int]:
        """Get the qubit indices that encode atom existence for a given atom.
        
        For the dynamic circuit, atoms are encoded in 2 qubits. The existence
        of an atom is determined by whether any of these 2 qubits is 1.
        
        In this dynamic circuit design, qubits 2 and 3 are REUSED for encoding
        different atoms after being reset via reset_previous_atom_bond_circuit().
        Therefore, the existence check for any atom (after the first two atoms)
        always uses qubits 2 and 3, regardless of heavy_atom_number.
        
        Args:
            heavy_atom_number: The atom number to check existence for.
                This parameter is accepted for interface consistency but
                the return value is always [2, 3] due to qubit reuse.
        
        Returns:
            List of qubit indices: always [2, 3] for this circuit design.
        """
        # Qubits 2,3 are reused for all atoms after reset; see circuit design docs
        return [2, 3]

    def build_atom_type_circuit(
        self, heavy_atom_number: int, weight_vector: Union[List[float], np.ndarray]
    ):
        """Build the circuit for atom type encoding.
        
        The branch is executed if the previous atom exists (any of its qubits is 1).
        This uses the ancilla flag pattern to satisfy c_if's single-handle requirement.
        """
        assert len(weight_vector) == 3
        qubit_1_index = 2
        qubit_2_index = 3
        clbit_1_index = (heavy_atom_number - 1) ** 2 + (heavy_atom_number - 1)
        clbit_2_index = clbit_1_index + 1

        # Get the qubit indices for previous atom existence check
        existence_qubits = self._atom_existence_qubit_indices(heavy_atom_number - 1)
        
        # Compute OR condition before measurement
        flag_idx = self._compute_or_condition(existence_qubits)
        flag_handle = self.kernel.mz(self.qubits[flag_idx], "_flag")

        def atom_branch():
            self.kernel.ry(np.pi * weight_vector[0], self.qubits[qubit_1_index])
            self.kernel.ry(np.pi * weight_vector[1], self.qubits[qubit_2_index])
            self.controlled_ry(qubit_1_index, qubit_2_index, weight_vector[2])

        self.kernel.c_if(flag_handle, atom_branch)
        self._measure_qubits(
            [qubit_1_index, qubit_2_index], [clbit_1_index, clbit_2_index]
        )

    def build_bond_type_circuit(
        self,
        heavy_atom_number: int,
        fixed_weight_vector: Union[List[float], np.ndarray],
        flexible_weight_vector: Union[List[float], np.ndarray],
    ):
        """Build the circuit for bond type encoding.
        
        This uses two conditional checks:
        1. Atom existence: Execute bond logic only if the current atom exists
        2. Bond presence: Apply "toggle then conditionally untoggle" pattern
           to ensure the last qubit is set when no bond is present
        
        Toggle-then-conditionally-untoggle Pattern:
        ------------------------------------------
        This pattern is equivalent to Qiskit's if_test((register, 0)) where we
        want to apply an X gate only if all measured qubits are 0:
        
        1. Apply X to target qubit (toggle to 1)
        2. For each bond qubit: if measured as 1, apply X to untoggle
        
        The result: target = 1 if all bonds are 0 (no bond exists),
                    target = 0 if any bond is 1 (bond exists)
        
        Note: This uses individual c_if calls per measurement handle,
        which conforms to CUDA-Q's single-handle requirement.
        """
        assert len(fixed_weight_vector) == heavy_atom_number - 1
        assert len(flexible_weight_vector) == 2 * (heavy_atom_number - 1)
        qubit_start_index = 4
        qubit_end_index = qubit_start_index + 2 * (heavy_atom_number - 1)
        clbit_start_index = heavy_atom_number ** 2 - heavy_atom_number + 2
        clbit_end_index = clbit_start_index + 2 * (heavy_atom_number - 1)

        # Check atom existence
        existence_qubits = self._atom_existence_qubit_indices(heavy_atom_number)
        flag_idx = self._compute_or_condition(existence_qubits)
        existence_handle = self.kernel.mz(self.qubits[flag_idx], "_flag")

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

        self.kernel.c_if(existence_handle, bond_branch)

        # Measure bond qubits to a temporary condition register
        bond_qubit_indices = list(range(qubit_start_index, qubit_end_index))
        bond_clbit_indices = list(range(clbit_start_index, clbit_end_index))
        cond_handles = self._measure_qubits(
            bond_qubit_indices,
            bond_clbit_indices,
            register_name="_cond",
        )

        target_index = qubit_end_index - 1
        # Toggle: Apply X to set target to 1 (assuming no bond)
        self.kernel.x(self.qubits[target_index])
        
        # Conditionally untoggle: For each bond measurement, if it's 1,
        # untoggle (apply X) to set target back to 0.
        # This implements: target = 1 if all bonds are 0, else target = 0
        # Equivalent to original Qiskit if_test((register, 0)) behavior.
        for handle in cond_handles:
            self.kernel.c_if(
                handle, lambda idx=target_index: self.kernel.x(self.qubits[idx])
            )

        # Measure to main register
        self._measure_qubits(
            bond_qubit_indices,
            bond_clbit_indices,
        )

    def _measure_all_qubits(self):
        """Measure all qubits to the auxiliary register.
        
        This is done to prevent CUDA-Q from eliding any qubits. The auxiliary
        register "_all" contains all qubit states but is NOT used for the
        primary output - use get_register_counts(main_register) instead.
        """
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
