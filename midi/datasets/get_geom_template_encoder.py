import os
from glob import glob
import numpy as np
from rdkit import Chem, RDLogger
import pickle
import torch
from torch_geometric.data.collate import collate
import midi.datasets.dataset_utils as dataset_utils
from midi.metrics.metrics_utils import compute_all_statistics
from torch import Tensor
from torch_geometric.data import Batch, Data
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)

def my_collate(
        data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
    r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
    to the internal storage format of
    :class:`~torch_geometric.data.InMemoryDataset`."""
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices


remove_h = False
h = 'noh' if remove_h else 'h'

# base_dir = '/mnt/home/linjie/projects/diffusion_model/data/geom_drug_data/CASF-2016'
# ligand_paths = glob(f"{base_dir}/*/*_ligand.sdf")
# save_dir = '/mnt/home/linjie/projects/diffusion_model/data/geom_drug_data/geom_with_h_CASF2016/processed'
# os.makedirs(save_dir)
sdf_dir = '/mnt/home/linjie/projects/diffusion_model/data/geom_drug_data/geom_low1/raw/geom_drug_sdf'
base_dir = '/mnt/home/linjie/projects/diffusion_model/data/geom_drug_data/geom_with_h_template'
save_dir = f"{base_dir}/processed"
with open(f"{base_dir}/raw/geom_testset_noH_filter_filename100.txt")as f:
    filenames = f.read().splitlines()
data_name = 'template100'
template_mols = []
for name in filenames:
    template_mols.append(Chem.SDMolSupplier(f"{sdf_dir}/{name}", removeHs=False)[0])


template_data = []
all_smiles = []
for idx, ligand_mol in enumerate(template_mols):
    processed_paths = [f'test_{h}_{data_name}.pt', f'test_n_{h}_{data_name}.pickle', f'test_atom_types_{h}_{data_name}.npy', f'test_bond_types_{h}_{data_name}.npy',
                       f'test_charges_{h}_{data_name}.npy', f'test_valency_{h}_{data_name}.pickle', f'test_bond_lengths_{h}_{data_name}.pickle',
                       f'test_angles_{h}_{data_name}.npy', f'test_smiles_{data_name}.pickle', f'test_template_{h}_{data_name}.pt']
    ligand_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(ligand_mol)))
    # print(ligand_smiles)
    all_smiles.append(ligand_smiles)
    data = dataset_utils.mol_to_torch_geometric(ligand_mol, full_atom_encoder, ligand_smiles)
    if remove_h:
        data = dataset_utils.remove_hydrogens(data)
    # data.idx = ligand_mol.GetProp('template_idx').split('template')[-1]
    # data.idx = ligand_mol.GetProp('template_idx')
    data.idx = str(idx)
    data.id = ligand_mol.GetProp('_Name')
    template_data.append(data)

os.chdir(save_dir)
torch.save(my_collate(template_data), processed_paths[0])
torch.save(template_data, processed_paths[9])

statistics = compute_all_statistics(template_data, full_atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5})
save_pickle(statistics.num_nodes, processed_paths[1])
np.save(processed_paths[2], statistics.atom_types)
np.save(processed_paths[3], statistics.bond_types)
np.save(processed_paths[4], statistics.charge_types)
save_pickle(statistics.valencies, processed_paths[5])
save_pickle(statistics.bond_lengths, processed_paths[6])
np.save(processed_paths[7], statistics.bond_angles)
save_pickle(set(all_smiles), processed_paths[8])

with open(f"{base_dir}/raw/test_data_{data_name}.txt", 'w')as f:
    for smiles in all_smiles:
        f.write(smiles+'\n')
