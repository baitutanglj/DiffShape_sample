import os

import numpy as np
import torch
import torch.nn.functional as F

import midi.datasets.dataset_utils as dataset_utils
from midi.datasets.abstract_dataset import AbstractDatasetInfos
from midi.utils import PlaceHolder

full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}

class GeomInfos(AbstractDatasetInfos):
    def __init__(self,):
        self.need_to_strip = False
        self.remove_h = False
        # self.datadir = datadir
        self.h = 'noh' if self.remove_h else 'h'# to indicate whether we need to ignore one output from the model

        self.statistics = {**self.load_statistics('train')}
        self.name = 'geom'
        self.atom_encoder = full_atom_encoder
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().complete_infos(self.statistics, self.atom_encoder)
        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=0, pos=3)

    def to_one_hot(self, X, charges, E, node_mask, just_control=False):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask, just_control)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()


    def load_statistics(self, data_type):
        # processed_name = [f'{data_type}_{self.h}.pt', f'{data_type}_n_{self.h}.pickle', f'{data_type}_atom_types_{self.h}.npy',
        #                   f'{data_type}_bond_types_{self.h}.npy',
        #                   f'{data_type}_charges_{self.h}.npy', f'{data_type}_valency_{self.h}.pickle',
        #                   f'{data_type}_bond_lengths_{self.h}.pickle',
        #                   f'{data_type}_angles_{self.h}.npy', f'{data_type}_smiles.pickle', f'{data_type}_template_{self.h}.pt']
        # self.processed_paths = [os.path.join(self.datadir, name) for name in processed_name]
        # data_type_statistics = dataset_utils.Statistics(
        #     atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
        #     bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
        #     charge_types=torch.from_numpy(np.load(self.processed_paths[4])))

        atom_types = np.array([4.4053e-01, 9.3172e-07, 4.0611e-01, 6.4814e-02, 6.6239e-02, 4.8541e-03,
                               0.0000e+00, 9.1271e-07, 1.0715e-04, 1.2214e-02, 4.0416e-03, 0.0000e+00,
                               1.0664e-03, 1.8977e-05, 0.0000e+00, 7.6059e-08])
        bond_types = np.array([9.5518e-01, 3.0673e-02, 2.0289e-03, 4.5172e-05, 1.2072e-02])
        charge_types = np.array([[0.0000e+00, 0.0000e+00, 1.0000e+00, 2.4172e-06, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 2.2449e-01, 7.7551e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 7.8941e-05, 9.9988e-01, 3.9049e-05, 0.0000e+00, 0.0000e+00],
                                 [1.7603e-05, 6.3663e-05, 9.6235e-01, 3.7567e-02, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 3.5188e-02, 9.6479e-01, 2.4400e-05, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 3.9173e-06, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 0.0000e+00, 7.9167e-01, 2.0833e-01, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 7.0985e-04, 8.3088e-01, 1.6841e-01, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 2.3352e-05, 9.3405e-01, 6.5722e-02, 2.8022e-05, 1.7436e-04],
                                 [0.0000e+00, 2.3524e-05, 9.9996e-01, 1.4114e-05, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 0.0000e+00, 9.9971e-01, 2.8529e-04, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 0.0000e+00, 9.9699e-01, 0.0000e+00, 3.0060e-03, 0.0000e+00],
                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                 [0.0000e+00, 0.0000e+00, 2.5000e-01, 0.0000e+00, 7.5000e-01, 0.0000e+00]])
        data_type_statistics = dataset_utils.Statistics(
            atom_types=torch.from_numpy(atom_types),
            bond_types=torch.from_numpy(bond_types),
            charge_types=torch.from_numpy(charge_types))

        return {data_type: data_type_statistics}

