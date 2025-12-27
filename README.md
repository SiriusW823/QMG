# QMG: Quantum-based Molecule Generator
This repository demonstrates the useage of quantum circuits for the generation of small molecules.

The building blocks of the quantum circuits are illustrated in the figure below:

<img src="docs/Figure_1.svg" style="width: 100%; height: auto;">

## OS Requirements
This repository requires to operate on **Linux** operating system.

## Python Dependencies
* Python (version >= 3.12)
* cudaq (version == 0.12.0)
* rdkit (version >= 2024.3.5)
* matplotlib (version >=3.9.2)
* pylatexenc (version >= 2.10)
* numpy (version >= 2.1.1)
* pandas (version >= 2.2.3)

## Example script for unconditional generation of small molecules (with number of heavy atoms <=5).

```python
from qmg.generator import MoleculeGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# parameter settings
num_heavy_atom = 5
num_sample = 10000
random_seed = 7

mg = MoleculeGenerator(num_heavy_atom) 
smiles_dict, validity, diversity = mg.sample_molecule(num_sample, ransom_seed)
print(smiles_dict)
print("Validity: {:.2f}%".format(validity*100))
print("Diversity: {:.2f}%".format(diversity*100))

# Example outputs:
# {'NO': 915, None: 695, 'CC': 1659, 'O': 288, 'CN': 886, 'C': 3427, 'NCNCN': 51, 'CNNNN': 1, 'N[C@H]1CCN1': 37, ...
# Validity: 93.05%
# Diversity: 2.75%
```


## Example script for conditional generation of small molecules with epoxide substructure (with number of heavy atoms <=7).

```python
from qmg.generator import MoleculeGenerator
from qmg.utils import ConditionalWeightsGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# parameter settings
num_heavy_atom = 7
random_seed = 3
smarts = "[O:1]1[C:2][C:3]1"
disable_connectivity_position = [1]
num_sample = 10000

cwg = ConditionalWeightsGenerator(num_heavy_atom, smarts=smarts, disable_connectivity_position=disable_connectivity_position)
random_weight_vector = cwg.generate_conditional_random_weights(random_seed)
mg = MoleculeGenerator(num_heavy_atom, all_weight_vector=random_weight_vector) 
smiles_dict, validity, diversity = mg.sample_molecule(num_sample)
print(smiles_dict)
print("Validity: {:.2f}%".format(validity*100))
print("Diversity: {:.2f}%".format(diversity*100))

# Example outputs:
# {None: 3882, 'CON[C@@]1(N)CO1': 111, 'NCNN[C@@H]1CO1': 47, 'ON[C@@H]1CO1': 946, 'NON[C@@]1(N)CO1': 246, 'NON[C@H]1O[C@@H]1N': 33, 'NNON[C@@H]1CO1': 246, ...
# Validity: 61.18%
# Diversity: 6.12%
```
