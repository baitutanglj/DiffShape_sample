from rdkit import Chem
from rdkit.Chem import QED
import os
from collections import Counter
import math
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import sys
from torchmetrics import MeanMetric, MaxMetric

from midi.utils import NoSyncMetric as Metric, NoSyncMetricCollection as MetricCollection
from midi.analysis.rdkit_functions import check_stability
from midi.utils import NoSyncMAE as MeanAbsoluteError
from midi.metrics.metrics_utils import counter_to_tensor, wasserstein1d, total_variation1d
from collections import defaultdict
from midi.utils import get_template_sdf

filter_substructure = ["[*;r8]",
                       "[*;r9]",
                       "[*;r10]",
                       "[*;r11]",
                       "[*;r12]",
                       "[*;r13]",
                       "[*;r14]",
                       "[*;r15]",
                       "[*;r16]",
                       "[*;r17]",
                       "[#8][#8]",
                       "[#6;+]",
                       "[#16][#16]",
                       "[#7;!n][S;!$(S(=O)=O)]",
                       "[#7;!n][#7;!n]",
                       "C#C",
                       "C(=[O,S])[O,S]",
                       "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                       "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                       "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                       "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                       "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                       "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
                       "[*]1[*]=,:[*]1",
                       "[*]1[*]=,:[*]1=[*]",
                       "[N]=[S]=[N]"]

class SamplingMetrics(nn.Module):
    def __init__(self, train_smiles, dataset_infos, test, template=None, test_smiles=None, filter=True):
        super().__init__()
        self.filter_smarts = [Chem.MolFromSmarts(subst) for subst in filter_substructure if Chem.MolFromSmarts(subst)]
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder
        self.filter = filter
        self.train_smiles = train_smiles
        self.test_smiles = set(test_smiles) if test_smiles is not None else None
        self.test = test
        self.template, _, = get_template_sdf(template, dataset_infos) if template else None
        self.template_dict = {i.GetProp('template_idx'): i for i in self.template}
        self.atom_stable = MeanMetric()
        self.mol_stable = MeanMetric()

        # Retrieve dataset smiles only for qm9 currently.
        self.train_smiles = set(train_smiles) if train_smiles is not None else None
        self.validity_metric = MeanMetric()
        self.uniqueness = MeanMetric()
        self.novelty = MeanMetric()
        self.test_smiles_novelty = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.atom_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()
        self.charge_w1 = MeanMetric()
        self.valency_w1 = MeanMetric()
        self.bond_lengths_w1 = MeanMetric()
        self.angles_w1 = MeanMetric()
        self.validity_flag = True

    def reset(self):
        for metric in [self.atom_stable, self.mol_stable, self.validity_metric, self.uniqueness,
                       self.novelty, self.test_smiles_novelty, self.mean_components, self.max_components, self.num_nodes_w1,
                       self.atom_types_tv, self.edge_types_tv, self.charge_w1, self.valency_w1,
                       self.bond_lengths_w1, self.angles_w1]:
            metric.reset()

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        all_valid_mols = defaultdict(list)
        error_message = Counter()
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(largest_mol)
                    if self.filter:
                        match = any([largest_mol.HasSubstructMatch(subst) for subst in self.filter_smarts])
                        if match:
                            continue
                    smiles = Chem.MolToSmiles(largest_mol)
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    largest_mol.SetProp('smiles', smiles)
                    largest_mol.SetProp('template_idx', rdmol.GetProp('template_idx'))
                    largest_mol.SetProp('qed', str(QED.qed(largest_mol)))
                    all_valid_mols[rdmol.GetProp('template_idx')].append(largest_mol)
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    # sys.stdout.write("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    # sys.stdout.write("Can't kekulize molecule")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
                except:
                    error_message[3] += 1
        sys.stdout.write(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {error_message[-1]}\n")
        sys.stdout.flush()
        self.validity_metric.update(value=len(valid) / len(generated), weight=len(generated))
        num_components = torch.tensor(num_components, device=self.mean_components.device)
        self.mean_components.update(num_components)
        self.max_components.update(num_components)
        not_connected = 100.0 * error_message[4] / len(generated)
        connected_components = 100.0 - not_connected
        return valid, connected_components, all_smiles, all_valid_mols, error_message

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        # Validity
        valid, connected_components, all_smiles, all_valid_mols, error_message = self.compute_validity(generated)

        validity = self.validity_metric.compute()
        uniqueness, novelty = torch.tensor(0.0), torch.tensor(0.0)
        test_smiles_novelty = 0
        mean_components = self.mean_components.compute()
        max_components = self.max_components.compute()

        # Uniqueness
        if len(valid) > 0:
            unique = list(set(valid))
            self.uniqueness.update(value=len(unique) / len(valid), weight=len(valid))
            uniqueness = self.uniqueness.compute()

            # if self.train_smiles is not None:
            #     novel = []
            #     for smiles in unique:
            #         if smiles not in self.train_smiles:
            #             novel.append(smiles)
            #     self.novelty.update(value=len(novel) / len(unique), weight=len(unique))
            #     novelty = self.novelty.compute()

            if self.test_smiles is not None:
                novel = []
                for smiles in unique:
                    if smiles not in self.test_smiles:
                        novel.append(smiles)
                self.test_smiles_novelty.update(value=len(novel) / len(unique), weight=len(unique))
                test_smiles_novelty = self.test_smiles_novelty.compute()

        num_molecules = int(self.validity_metric.weight.item())
        sys.stdout.write(f"Validity over {num_molecules} molecules:"
              f" {validity * 100 :.2f}%\n")
        sys.stdout.write(f"Number of connected components of {num_molecules} molecules: "
              f"mean:{mean_components:.2f} max:{max_components:.2f}\n")
        sys.stdout.write(f"Connected components of {num_molecules} molecules: "
              f"{connected_components:.2f}\n")
        sys.stdout.write(f"Uniqueness: {uniqueness * 100 :.2f}% WARNING: do not trust this metric on multi-gpu\n")
        # sys.stdout.write(f"Novelty: {novelty * 100 :.2f}%")
        sys.stdout.write(f"Validity * Uniqueness: {validity*uniqueness*100 :.2f}%\n")
        sys.stdout.write(f"test_smiles_novelty: {test_smiles_novelty * 100 :.2f}%\n")
        sys.stdout.flush()

        result_dict = {
            'sample_num_count': num_molecules,
            'Validity': validity.item()*100,
            'Uniqueness': uniqueness.item()*100,
            'Validity*Uniqueness': (validity * uniqueness).item()*100,
            'validity_num': self.validity_metric.value.item(),
            'uniqueness_num': self.uniqueness.value.item(),
            # 'Novelty': novelty.item()*100,
            'Connected_Components': connected_components
        }

        return all_smiles, all_valid_mols, result_dict

    def __call__(self, molecules: list, name, result_path):
        # Atom and molecule stability
        for i, mol in enumerate(molecules):
            mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_infos)
            self.mol_stable.update(value=mol_stable)
            self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

        stability_dict = {'mol_stable': self.mol_stable.compute().item(),
                          'atom_stable': self.atom_stable.compute().item()}
        sys.stdout.write(f"Stability metrics: {stability_dict}")
        sys.stdout.write('\n')
        sys.stdout.flush()

        # Validity, uniqueness, novelty
        all_generated_smiles, all_valid_mols, result_dict = self.evaluate(molecules)
        result_dict.update(stability_dict)
        result_df = pd.DataFrame(result_dict, index=[0])
        # os.makedirs(result_path, exist_ok=True)

        if len(all_valid_mols.keys())>0:
            self.validity_flag = True
            for idx in all_valid_mols.keys():
                result_df.insert(0,'template_idx',idx)
                # smiles_set = set()
                # sample_i = 0
                # template = self.template_dict.get(idx)
                # # smiles_f = open(f"{filename}_template{idx}.txt", 'w')
                # with Chem.SDWriter(result_path+f"{idx}.sdf")as f:
                #     f.write(template)
                #     for i, mol in enumerate(all_valid_mols[idx]):
                #         if mol.GetProp("smiles") not in smiles_set:
                #             mol.SetProp('_Name', f"sample{sample_i}")
                #             f.write(mol)
                #             # smiles_f.write(mol.GetProp('smiles')+'\n')
                #             sample_i +=1
                #             smiles_set.add(mol.GetProp("smiles"))

                # smiles_f.close()
            df_path = result_path+"result_metric.csv"
            result_df.to_csv(df_path, mode='a', index=False, header=False, float_format='%.2f')
            self.reset()

        else:
            self.validity_flag = False
            self.reset()


def number_nodes_distance(molecules, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for molecule in molecules:
        c[molecule.num_nodes] += 1

    generated_n = counter_to_tensor(c)
    return wasserstein1d(generated_n, reference_n)


def atom_types_distance(molecules, target, save_histogram=False):
    generated_distribution = torch.zeros_like(target)
    for molecule in molecules:
        for atom_type in molecule.atom_types:
            generated_distribution[atom_type] += 1
    # if save_histogram:
    #     np.save('generated_atom_types.npy', generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def bond_types_distance(molecules, target, save_histogram=False):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        bond_types = molecule.bond_types
        mask = torch.ones_like(bond_types)
        mask = torch.triu(mask, diagonal=1).bool()
        bond_types = bond_types[mask]
        unique_edge_types, counts = torch.unique(bond_types, return_counts=True)
        for type, count in zip(unique_edge_types, counts):
            generated_distribution[type] += count
    # if save_histogram:
    #     np.save('generated_bond_types.npy', generated_distribution.cpu().numpy())
    sparsity_level = generated_distribution[0] / torch.sum(generated_distribution)
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class, sparsity_level


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.atom_types == atom_type
            if mask.sum() > 0:
                at_charges = dataset_infos.one_hot_charges(molecule.charges[mask])
                generated_distribution[atom_type] += at_charges.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities.to(device)).item()

    return w1, w1_per_class


def valency_distance(molecules, target_valencies, atom_types_probabilities, atom_encoder):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values())
    max_valency_generated = max(max(vals.keys()) if len(vals) > 0 else -1 for vals in generated_valencies.values())
    max_valency = max(max_valency_target, max_valency_generated)

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[atom_encoder[atom_type], valency] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[atom_type, valency] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


def bond_length_distance(molecules, target, bond_types_probabilities):
    generated_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for molecule in molecules:
        cdists = torch.cdist(molecule.positions.unsqueeze(0),
                             molecule.positions.unsqueeze(0)).squeeze(0)
        for bond_type in range(1, 5):
            edges = torch.nonzero(molecule.bond_types == bond_type)
            bond_distances = cdists[edges[:, 0], edges[:, 1]]
            distances_to_consider = torch.round(bond_distances, decimals=2)
            for d in distances_to_consider:
                generated_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, 5):
        s = sum(generated_bond_lenghts[bond_type].values())
        if s == 0:
            s = 1
        for d, count in generated_bond_lenghts[bond_type].items():
            generated_bond_lenghts[bond_type][d] = count / s

    # Convert both dictionaries to tensors
    min_generated_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in generated_bond_lenghts.values())
    min_target_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in target.values())
    min_length = min(min_generated_length, min_target_length)

    max_generated_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in generated_bond_lenghts.values())
    max_target_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in target.values())
    max_length = max(max_generated_length, max_target_length)

    num_bins = int((max_length - min_length) * 100) + 1
    generated_bond_lengths = torch.zeros(4, num_bins)
    target_bond_lengths = torch.zeros(4, num_bins)

    for bond_type in range(1, 5):
        for d, count in generated_bond_lenghts[bond_type].items():
            bin = int((d - min_length) * 100)
            generated_bond_lengths[bond_type - 1, bin] = count
        for d, count in target[bond_type].items():
            bin = int((d - min_length) * 100)
            target_bond_lengths[bond_type - 1, bin] = count

    cs_generated = torch.cumsum(generated_bond_lengths, dim=1)
    cs_target = torch.cumsum(target_bond_lengths, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 100    # 100 because of bin size
    weighted = w1_per_class * bond_types_probabilities[1:]
    return torch.sum(weighted).item(), w1_per_class


def angle_distance(molecules, target_angles, atom_types_probabilities, valencies, atom_decoder, save_histogram: bool):
    num_atom_types = len(atom_types_probabilities)
    generated_angles = torch.zeros(num_atom_types, 180 * 10 + 1)
    for molecule in molecules:
        adj = molecule.bond_types
        pos = molecule.positions
        for atom in range(adj.shape[0]):
            p_a = pos[atom]
            neighbors = torch.nonzero(adj[atom]).squeeze(1)
            for i in range(len(neighbors)):
                p_i = pos[neighbors[i]]
                for j in range(i + 1, len(neighbors)):
                    p_j = pos[neighbors[j]]
                    v1 = p_i - p_a
                    v2 = p_j - p_a
                    assert not torch.isnan(v1).any()
                    assert not torch.isnan(v2).any()
                    prod = torch.dot(v1 / (torch.norm(v1) + 1e-6), v2 / (torch.norm(v2) + 1e-6))
                    if prod > 1:
                        sys.stdout.write(f"Invalid angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}\n")
                        sys.stdout.flush()
                    prod.clamp(min=0, max=1)
                    angle = torch.acos(prod)
                    if torch.isnan(angle).any():
                        sys.stdout.write(f"Nan obtained in angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}\n")
                        sys.stdout.flush()
                    else:
                        bin = int(torch.round(angle * 180 / math.pi, decimals=1).item() * 10)
                        generated_angles[molecule.atom_types[atom], bin] += 1

    s = torch.sum(generated_angles, dim=1, keepdim=True)
    s[s == 0] = 1
    generated_angles = generated_angles / s
    # if save_histogram:
    #     np.save('generated_angles_historgram.npy', generated_angles.numpy())

    if type(target_angles) in [np.array, np.ndarray]:
        target_angles = torch.from_numpy(target_angles).float()

    cs_generated = torch.cumsum(generated_angles, dim=1)
    cs_target = torch.cumsum(target_angles, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    # The atoms that have a valency less than 2 should not matter
    valency_weight = torch.zeros(len(w1_per_type), device=w1_per_type.device)
    for i in range(len(w1_per_type)):
        valency_weight[i] = 1 - valencies[atom_decoder[i]][0] - valencies[atom_decoder[i]][1]

    weighted = w1_per_type * atom_types_probabilities * valency_weight
    return (torch.sum(weighted) / (torch.sum(atom_types_probabilities * valency_weight) + 1e-5)).item(), w1_per_type



class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """ Compute the distance between histograms. """
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples



class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state('total_edge', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples



class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class AlCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AsCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class HgCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {'H': HydrogenCE, 'B': BoronCE, 'C': CarbonCE, 'N': NitroCE, 'O': OxyCE, 'F': FluorCE,
                      'Al': AlCE, 'Si': SiCE, 'P': PhosphorusCE, 'S': SulfurCE, 'Cl': ClCE, 'As': AsCE,
                      'Br': BrCE,  'I': IodineCE, 'Hg': HgCE, 'Bi': BiCE}

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])

