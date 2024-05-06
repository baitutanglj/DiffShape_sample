import itertools
import os
import time
from collections import defaultdict
from typing import List
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Geometry import Point3D
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from copy import deepcopy
# from midi.datasets import geom_dataset
from midi import utils
from midi.analysis.rdkit_functions import Molecule
from midi.control_model import ControlNet, ControlGraphTransformer
from midi.diffusion.extra_features import ExtraFeatures
# print("RUNNING ABLATION")
from midi.diffusion.noise_model import DiscreteUniformTransition, MarginalUniformTransition
from midi.metrics.molecular_metrics import SamplingMetrics
from midi.metrics.molecular_metrics import filter_substructure
from midi.utils import PlaceHolder
from midi.utils import get_template_sdf


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class GateResidue(nn.Module):
    def __init__(self, input_dims: utils.PlaceHolder, full_gate: bool = True):
        super(GateResidue, self).__init__()
        self.input_dims = input_dims
        if full_gate:
            self.gate_X = torch.nn.Linear((input_dims.X + input_dims.charges) * 3, input_dims.X + input_dims.charges)
            self.gate_E = torch.nn.Linear(input_dims.E * 3, input_dims.E)
            self.gate_pos = torch.nn.Linear(input_dims.pos * 3, input_dims.pos)
            # self.gate_y = torch.nn.Linear(input_dims.y * 3, input_dims.y)
        else:
            self.gate_X = torch.nn.Linear(input_dims.X * 3, 1)
            self.gate_E = torch.nn.Linear(input_dims.E * 3, 1)
            self.gate_pos = torch.nn.Linear(input_dims.pos * 3, 1)
            # self.gate_y = torch.nn.Linear(input_dims.y * 3, 1)

    def forward(self, x, res):
        x_X_tmp = torch.cat((x.X, x.charges), dim=-1)  # torch.Size([20, 64, 22])
        res_X_tmp = torch.cat((res.X, res.charges), dim=-1)
        g_X = self.gate_X(torch.cat((
            x_X_tmp,
            res_X_tmp,
            x_X_tmp - res_X_tmp), dim=-1)).sigmoid()  # torch.Size([20, 64, 22])
        g_E = self.gate_E(
            torch.cat((x.E, res.E, x.E - res.E), dim=-1)).sigmoid()  # x.E.shape:torch.Size([20, 64, 64, 5])
        g_pos = self.gate_pos(torch.cat((x.pos, res.pos, x.pos - res.pos), dim=-1)).sigmoid()  # torch.Size([20, 64, 3])
        # g_y = self.gate_y(torch.cat((x.y, res.y, x.y - res.y), dim=-1)).sigmoid()

        X = x_X_tmp * g_X + res_X_tmp * (1 - g_X)
        E = x.E * g_E + res.E * (1 - g_E)
        pos = x.pos * g_pos + res.pos * (1 - g_pos)
        E = 1 / 2 * (E + torch.transpose(E, 1, 2))
        out = utils.PlaceHolder(X=X[..., :self.input_dims.X], charges=X[..., self.input_dims.X:],
                                E=E, pos=pos, y=res.y, node_mask=res.node_mask).mask()
        return out


# class FullDenoisingDiffusion(pl.LightningModule):
class FullDenoisingDiffusion(nn.Module):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None

    def __init__(self, cfg, dataset_infos):
        super().__init__()
        self.filter_smarts = [Chem.MolFromSmarts(subst) for subst in filter_substructure if Chem.MolFromSmarts(subst)]
        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps
        self.condition_control = cfg.model.condition_control if hasattr(self.cfg.model, "condition_control") else False
        self.only_last_control = cfg.model.only_last_control if hasattr(self.cfg.model, "only_last_control") else False
        self.guess_mode = cfg.model.guess_mode
        self.control_scales = [cfg.model.strength * (0.825 ** float(12 - i)) for i in range(13)]
        self.unconditional_guidance_scale = cfg.model.unconditional_guidance_scale
        self.control_data_dict = cfg.dataset.control_data_dict if hasattr(self.cfg.dataset, "control_data_dict") else {
            'cX': 'cX', 'cE': 'cE', 'cpos': 'cpos'}
        self.control_add_noise_dict = cfg.dataset.control_add_noise_dict if hasattr(self.cfg.dataset,
                                                                                    "control_add_noise_dict") else {
            'cX': False, 'cE': False, 'cpos': False}
        self.add_gru_output_model = cfg.model.add_gru_output_model if hasattr(self.cfg.model,
                                                                              "add_gru_output_model") else False
        self.dropout_rate = cfg.model.dropout_rate if hasattr(self.cfg.model, "dropout_rate") else 0
        self.noise_std = cfg.model.noise_std if hasattr(self.cfg.model, "noise_std") else 0.3
        self.dataset_infos = dataset_infos
        self.extra_features = ExtraFeatures(cfg.model.extra_features, max_n_nodes=165)
        self.input_dims = self.extra_features.update_input_dims(dataset_infos.input_dims)
        self.output_dims = dataset_infos.output_dims

        self.control_model = ControlNet(
            input_dims=self.input_dims,
            n_layers=cfg.model.n_layers,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            dropout_rate=self.dropout_rate
        )
        self.model = ControlGraphTransformer(input_dims=self.input_dims,
                                             n_layers=cfg.model.n_layers,
                                             hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                             hidden_dims=cfg.model.hidden_dims,
                                             output_dims=self.output_dims)

        if self.add_gru_output_model:
            self.output_model = GateResidue(input_dims=self.output_dims, full_gate=True)

        self.instantiate_model_stage()

        if cfg.model.transition == 'uniform':
            self.noise_model = DiscreteUniformTransition(output_dims=self.output_dims,
                                                         cfg=cfg)
        elif cfg.model.transition == 'marginal':
            self.noise_model = MarginalUniformTransition(x_marginals=self.dataset_infos.atom_types,
                                                         e_marginals=self.dataset_infos.edge_types,
                                                         charges_marginals=self.dataset_infos.charges_marginals,
                                                         y_classes=self.output_dims.y,
                                                         cfg=cfg)
        else:
            assert ValueError(f"Transition type '{cfg.model.transition}' not implemented.")

        self.log_every_steps = cfg.general.log_every_steps

    def instantiate_model_stage(self):
        self.model = self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

    def sample(self, sample_batch_size: int, samples_to_generate_start: int,
               sample_type: str, test_template: List, test_smiles: List,
               result_path: str, name: str, filter: bool) -> None:
        self.sample_type = sample_type
        self.result_path = result_path
        self.filter = filter
        self.device = next(self.model.parameters()).device
        self.test_template_dict = {i.idx: i for i in test_template}
        self.template_mol, _, = get_template_sdf(test_template, self.dataset_infos)
        self.template_dict = {i.GetProp('template_idx'): i for i in self.template_mol}
        self.samples_to_generate = samples_to_generate_start
        self.test_sampling_metrics = SamplingMetrics(None, self.dataset_infos, test=True,
                                                     template=test_template, test_smiles=test_smiles,
                                                     filter=self.filter).to(self.device)
        column_names = ['template_idx', 'sample_num_count', 'Validity', 'Uniqueness',
                        'Validity*Uniqueness', 'validity_num', 'uniqueness_num',
                        'Connected_Components', 'mol_stable', 'atom_stable']
        all_template_start_time = time.time()
        run_time_list = []
        self.sample_count = 0
        if self.sample_type == "everything":
            for template in test_template:
                template_idx = template.idx
                sys.stdout.write(f"template {template_idx} Sampling start on {self.device}\n")
                sys.stdout.flush()
                start = time.time()
                template = [template]
                uniqueness_dict = {str(i.idx): defaultdict(dict) for i in template}
                samples, uniqueness_dict = self.sample_n_graphs(samples_to_generate=samples_to_generate_start,
                                               sample_batch_size=sample_batch_size,
                                               test=True, template=template,
                                               uniqueness_dict=uniqueness_dict)

                sys.stdout.write("Computing sampling metrics...")
                sys.stdout.flush()
                self.test_sampling_metrics(samples, name, result_path)
                if self.test_sampling_metrics.validity_flag == True:
                    template_end_time = round(time.time() - start, 2)
                    run_time_list.append(template_end_time)
                    sys.stdout.write(f'Done. {template_idx} Sampling took {template_end_time:.2f} seconds\n')
                    sys.stdout.flush()
                sys.stdout.write('*' * 20)
                sys.stdout.write('\n')
                sys.stdout.flush()

        elif self.sample_type == "only_validity":
            uniqueness_dict = {str(i.idx): defaultdict(dict) for i in test_template}
            for template in test_template:
                template_idx = template.idx
                sys.stdout.write(f"template {template_idx} Sampling start on {self.device}\n")
                sys.stdout.flush()
                start = time.time()
                template = [template]
                all_samples = []
                sample_while_count = 0
                samples_to_generate = samples_to_generate_start
                samples_to_generate = 10 if samples_to_generate < 10 else samples_to_generate
                while len(template) > 0:
                    samples_to_generate = (samples_to_generate//sample_batch_size+1)*sample_batch_size
                    samples, uniqueness_dict = self.sample_n_graphs(samples_to_generate=samples_to_generate,
                                                                    sample_batch_size=sample_batch_size,
                                                                    test=True, template=template,
                                                                    uniqueness_dict=uniqueness_dict)
                    template, len_dict = self.check_sample_count(uniqueness_dict, resample_idx=template_idx)
                    sys.stdout.write(f"sample template: {len_dict}\n")
                    sys.stdout.flush()
                    all_samples.extend(samples)
                    sample_while_count += 1
                    samples_to_generate = min(sample_batch_size, samples_to_generate_start)
                    samples_to_generate = 10 if samples_to_generate < 10 else samples_to_generate
                    if sample_while_count > 100:
                        with open(self.result_path + f"sample_count.txt", 'w') as f:
                            f.write(f"{samples_to_generate_start}")
                        break
                sys.stdout.write(f"sample_batch_count:{sample_while_count}\n")
                sys.stdout.write("Computing sampling metrics...\n")
                sys.stdout.flush()
                self.test_sampling_metrics(all_samples, self.name, result_path=result_path)
                if self.test_sampling_metrics.validity_flag == True:
                    template_end_time = round(time.time() - start, 2)
                    run_time_list.append(template_end_time)
                    sys.stdout.write(f'Done. {template_idx} Sampling took {template_end_time:.2f} seconds\n')
                sys.stdout.write('*' * 20)
                sys.stdout.write('\n')
                sys.stdout.flush()

        result_metric_path = result_path + "result_metric.csv"
        if os.path.exists(result_metric_path):
            result_metric_df = pd.read_csv(result_metric_path, header=None, names=column_names)
            result_metric_df['run_time'] = run_time_list
            result_metric_df.to_csv(result_metric_path, index=False, float_format='%.2f')
        sys.stdout.write(f"all template Sampling took {time.time() - all_template_start_time:.2f} seconds\n")
        sys.stdout.write(f"Test ends.\n")
        sys.stdout.flush()

    @torch.no_grad()
    def sample_n_graphs(self, samples_to_generate: int, sample_batch_size: int, test: bool, template: list = None,
                        uniqueness_dict=None):
        if samples_to_generate <= 0:
            return []
        samples = []
        idx = template[0].idx
        centroid_pos = template[0].centroid_pos
        sys.stdout.write(f"{len(template)} template, sample_num:{samples_to_generate},"
                         f"sample_batch_size:{sample_batch_size}\n")
        sys.stdout.flush()
        template = Batch.from_data_list(sum([list(itertools.repeat(i, samples_to_generate)) for i in template], []))
        template_loader = DataLoader(template, sample_batch_size, shuffle=True)
        for i, template_batch in enumerate(template_loader):
            template_batch = template_batch.cuda(self.device)
            dense_data = utils.to_dense(template_batch, self.dataset_infos)
            dense_data.idx = template_batch.idx
            current_n_list = torch.unique(template_batch.batch, return_counts=True)[1]
            n_nodes = current_n_list
            samples.extend(self.sample_batch(n_nodes=n_nodes, batch_id=i, template=dense_data,
                                             save_final=len(current_n_list),
                                             test=test, centroid_pos=centroid_pos))
            uniqueness_dict = self.compute_uniqueness_dict(samples, uniqueness_dict)
            self.save_sample_mols({idx: uniqueness_dict.get(idx)})
            if self.sample_type == 'everything':
                self.sample_count += len(n_nodes)
                with open(self.result_path + f"sample_count.txt", 'w')as f:
                    f.write(f"{self.sample_count}")
            elif self.sample_type == 'only_validity':
                resample_template, len_dict = self.check_sample_count(uniqueness_dict, resample_idx=idx)
                with open(self.result_path + f"sample_count.txt", 'w') as f:
                    v_list = [len(v) if len(v)<self.samples_to_generate else self.samples_to_generate for v in uniqueness_dict.values()]
                    self.sample_count = sum(v_list)
                    f.write(f"{self.sample_count}")
                if len(resample_template)==0:
                    break

            sys.stdout.write(f"Sampling a batch finished with graphs:{len(n_nodes)}\n")
            sys.stdout.flush()
        return samples, uniqueness_dict

    def check_sample_count(self, uniqueness_dict, resample_idx=None):
        resample_dict = {}
        len_dict = {}
        for idx, sample_dict in uniqueness_dict.items():
            len_dict.update({idx: len(sample_dict)})
            if len(sample_dict) < self.samples_to_generate:
                resample_dict.update({idx: self.test_template_dict.get(idx)})

        if resample_idx is None:
            template_list = list(resample_dict.values())
        else:
            template_list = [resample_dict.get(resample_idx)] if resample_idx in resample_dict else []

        return template_list, len_dict

    def compute_uniqueness_dict(self, samples, uniqueness_dict):
        for mol in samples:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(largest_mol)
                    if self.filter:
                        match = any([largest_mol.HasSubstructMatch(subst) for subst in self.filter_smarts])
                        if match:
                            continue
                    smiles = Chem.MolToSmiles(largest_mol)
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                    largest_mol.SetProp('smiles', smiles)
                    largest_mol.SetProp('template_idx', rdmol.GetProp('template_idx'))
                    largest_mol.SetProp('qed', str(QED.qed(largest_mol)))
                    if smiles not in uniqueness_dict[rdmol.GetProp('template_idx')].keys():
                        uniqueness_dict[rdmol.GetProp('template_idx')][smiles] = largest_mol
                except:
                    continue
        return uniqueness_dict

    def save_sample_mols(self, uniqueness_dict):
        for idx, mol_dict in uniqueness_dict.items():
            if len(mol_dict)>0:
                template = self.template_dict.get(idx)
                with Chem.SDWriter(self.result_path + f"{idx}.sdf") as f:
                    f.write(template)
                    for i, mol_ in enumerate(mol_dict.values()):
                        mol = deepcopy(mol_)
                        mol.SetProp('_Name', f"sample{i}")
                        # pos = np.array(mol.GetConformers()[0].GetPositions())
                        # new_pos = pos + centroid_pos
                        # conf = Chem.Conformer(mol.GetNumAtoms())
                        # for atom_idx in range(mol.GetNumAtoms()):
                        #     conf.SetAtomPosition(atom_idx, Point3D(new_pos[atom_idx, 0], new_pos[atom_idx, 1],
                        #                                            new_pos[atom_idx, 2]))
                        # mol.RemoveAllConformers()
                        # mol.AddConformer(conf)
                        f.write(mol)


    @torch.no_grad()
    def sample_batch(self, n_nodes: list, batch_id: int = 0,
                     save_final: int = 0, test: bool = True, template: PlaceHolder = None,
                     centroid_pos=None):
        """
        :param batch_id: int
        :param n_nodes: list of int containing the number of nodes to sample for each graph
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        # sys.stdout.write(f"Sampling a batch with {len(n_nodes)} graphs.\n")
        sys.stdout.flush()
        assert save_final >= 0
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask, template=template, noise_std=self.noise_std)

        assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()

        # n_max = z_T.X.size(1)
        z_t = z_T
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T, 1 if test else self.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones((batch_size, 1), dtype=torch.long, device=z_t.X.device)

            z_s = self.sample_zs_from_zt(z_t=z_t, s_int=s_array)

            z_t = z_s

        # Sample final data
        sampled = z_t.collapse(self.dataset_infos.collapse_charges)
        X, charges, E, y, pos = sampled.X, sampled.charges, sampled.E, sampled.y, sampled.pos
        idx = sampled.idx

        molecule_list, molecules_visualize = [], []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n]
            charge_vec = charges[i, :n]
            edge_types = E[i, :n, :n]
            conformer = pos[i, :n]
            template_idx = idx[i]
            molecule_list.append(Molecule(atom_types=atom_types, charges=charge_vec,
                                          bond_types=edge_types, positions=conformer,
                                          atom_decoder=self.dataset_infos.atom_decoder,
                                          template_idx=template_idx, centroid_pos=centroid_pos))

        return molecule_list

    def sample_zs_from_zt(self, z_t, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_int)
        return z_s

    @property
    def BS(self):
        return self.cfg.train.batch_size

    def apply_model(self, model_input, condition_control):
        if condition_control:
            control_out = self.control_model(model_input)
            control_out = {ckey: control_out[ckey].mul_scales(scale) for ckey, scale in
                           zip(control_out, self.control_scales)}
            model_out = self.model(model_input, control_out)
        else:
            control_out = None
            model_out = self.model(model_input, control_out)

        return model_out

    def forward(self, z_t, extra_data):
        assert z_t.node_mask is not None
        model_input = z_t.copy()
        model_input.X = torch.cat((z_t.X, extra_data.X), dim=2).float()
        model_input.E = torch.cat((z_t.E, extra_data.E), dim=3).float()
        model_input.y = torch.hstack((z_t.y, extra_data.y, z_t.t)).float()
        model_t = self.apply_model(model_input, self.condition_control)
        model_uncond = self.apply_model(model_input, False)

        if self.add_gru_output_model == False:
            model_t = model_t.minus_scales(model_uncond, model_t.node_mask)
            model_t_scale = model_t.mul_scales(self.unconditional_guidance_scale)
            model_out = model_uncond.add_scales(model_t_scale, model_t_scale.node_mask)
        else:
            model_out = self.output_model(model_uncond, model_t)

        ## model_out = model_t
        return model_out

