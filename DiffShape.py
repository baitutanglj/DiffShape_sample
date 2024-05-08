# Do not move these imports, the order seems to matter
import argparse
import os
import time
import uuid
import sys
import torch
from rdkit import Chem

import midi.datasets.dataset_utils as dataset_utils
from midi.datasets import geom_dataset
from midi.diffusion_model import FullDenoisingDiffusion

# from midi.metrics.metrics_utils import compute_all_statistics

full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}
remove_h = False
h = 'noh' if remove_h else 'h'
# model_names = ["AtomBondFuzz_nstd0.3", "AtomFuzz_nstd0.3", "ColourPointCloud_nstd0.3",
#                    "ColourPointCloudSingle_nstd0.3", "NoFuzz_nstd0.3", "PointCloud_nstd0.2", "PointCloud_nstd0.3",
#                    "PointCloud_nstd0.25", "PointCloudSingle_dropout0.1_nstd0.3", "PointCloudSingle_nstd0.3",
#                    "PointCloudSingle_nstd0.4", "PointCloudSingle_nstd0.35"]
model_names = ["AtomBondFuzz_nstd0.3", "NoFuzz_nstd0.3", "PointCloud_nstd0.3", "PointCloudSingle_nstd0.3"]

def get_control_data_dict(model_type):
    if 'AtomBondFuzz_nstd' in model_type:
        control_data_dict = {'cX': 'cX', 'cE': 'cE', 'cpos': 'cpos'}
    elif 'AtomFuzz_nstd' in model_type:
        control_data_dict = {'cX': 'cX', 'cE': 'E', 'cpos': 'cpos'}
    elif 'ColourPointCloud_nstd' in model_type:
        control_data_dict = {'cX': 'X', 'cE': 'None', 'cpos': 'cpos'}
    elif 'ColourPointCloudSingle_nstd' in model_type:
        control_data_dict = {'cX': 'X', 'cE': 'single_mask_None', 'cpos': 'cpos'}
    elif 'NoFuzz_nstd' in model_type:
        control_data_dict = {'cX': 'X', 'cE': 'E', 'cpos': 'pos'}
    elif 'PointCloudSingle_nstd' in model_type:
        control_data_dict = {'cX': 'cX', 'cE': 'single_mask_None', 'cpos': 'cpos'}
    elif 'PointCloud_nstd' in model_type:
        control_data_dict = {'cX': 'cX', 'cE': 'None', 'cpos': 'cpos'}
    else:
        control_data_dict = {'cX': 'cX', 'cE': 'None', 'cpos': 'cpos'}

    return control_data_dict


def get_template(ligand_path, control_data_dict):
    ligand_mols = Chem.SDMolSupplier(ligand_path, removeHs=False)
    try:
        ligand_names = [mol.GetProp('_Name') for mol in ligand_mols]
    except:
        # basename = (os.path.basename(ligand_path)).split('.sdf')[0]
        ligand_names = [f"{str(i)}" for i in range(len(ligand_mols))]

    # ligand_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol))) for mol in ligand_mols]
    template_data, all_smiles = [], []
    for i, (mol, names) in enumerate(zip(ligand_mols, ligand_names)):
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
            data = dataset_utils.mol_to_torch_geometric(mol, full_atom_encoder, smiles)
            if remove_h:
                data = dataset_utils.remove_hydrogens(data)
            data.idx = names
            template_data.append(data)
            all_smiles.append(smiles)
            sys.stdout.write(f"{smiles}\n")
            sys.stdout.flush()
        except:
            sys.stdout.write(f"mol {i} in file is invalid\n")
            continue

    for i, _ in enumerate(template_data):
        template_data[i].cx = template_data[i].cx if control_data_dict['cX'] == 'cX' else \
        template_data[i].x
        template_data[i].ccharges = template_data[i].ccharges if control_data_dict['cX'] == 'cX' else \
        template_data[i].charges
        if control_data_dict['cE'] == 'cE':
            template_data[i].cedge_attr = template_data[i].cedge_attr
        elif control_data_dict['cE'] == 'None':
            template_data[i].cedge_attr = torch.zeros_like(template_data[i].cedge_attr)
        elif control_data_dict['cE'] == 'E':
            template_data[i].cedge_attr = template_data[i].edge_attr
        elif control_data_dict['cE'] == 'single_mask_None':
            mask_tensor = torch.rand(template_data[i].cedge_attr.shape[0]) > 0.5
            template_data[i].cedge_attr[mask_tensor] = 0

    return template_data, all_smiles


def define_model(ckpt_path, dataset_infos):
    ckpt_dict = torch.load(ckpt_path)
    cfg = ckpt_dict['hyper_parameters']['cfg']
    model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos)
    model.load_state_dict(ckpt_dict['state_dict'])
    return model


def load_all_model(model_dir, dataset_infos, model_name=None):
    global model_names
    model_dict = {}
    if model_name:
        model_dict.update(
            {model_name: define_model(f"{model_dir}/{model_name}.ckpt", dataset_infos=dataset_infos)})
    else:
        for model_name in model_names:
            model_dict.update(
                {model_name: define_model(f"{model_dir}/{model_name}.ckpt", dataset_infos=dataset_infos)})

    return model_dict


def main():
    parser = argparse.ArgumentParser(description="DiffShape sample molecular")
    parser.add_argument('-i', '--input', help='Sampling based on this input template molecule .sdf file', type=str)
    parser.add_argument('-o', '--output', help='Directory to save file of output molecule file', type=str)
    parser.add_argument('-b', '--batch_size', help='sample batch size', type=int, default=10)
    parser.add_argument('-n', '--sample_num', help='How many molecules to sample', type=int, default=3)
    parser.add_argument('-s', '--sample_type', help='"everything": Loop sampling n times,'
                                                    '"only_validity": Sample n molecules that are both valid and unique',
                        default='everything', choices=['everything', 'only_validity'])
    parser.add_argument('-m', '--model_type', help='Choose one of the models to sample', default='PointCloud_nstd0.2',
                        choices=model_names)
    parser.add_argument('-f', '--filter', help='Filter out molecules containing unreasonable substructures',
                        default='True', type=str)
    args = parser.parse_args()
    args.filter = args.filter == 'True'
    template_name = (os.path.basename(args.input)).split('.sdf')[0]
    now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    output = os.path.join(args.output, f"output_{now}_{str(uuid.uuid4())[:4]}_{args.model_type}_")
    os.makedirs(args.output, exist_ok=True)

    control_data_dict = get_control_data_dict(args.model_type)
    test_template, test_smiles = get_template(args.input, control_data_dict)

    # model_dir = './_internal/model'
    model_dir = f"{os.getcwd()}/data/model"
    dataset_infos = geom_dataset.GeomInfos()
    model_dict = load_all_model(model_dir, dataset_infos, model_name=args.model_type)
    full_model = model_dict[args.model_type]
    full_model.to("cuda:0")
    full_model.eval()
    # sys.stdout.write(full_model)
    with torch.no_grad():
        sample = full_model.sample(sample_batch_size=args.batch_size,
                                   samples_to_generate_start=args.sample_num,
                                   sample_type=args.sample_type,
                                   test_template=test_template,
                                   test_smiles=test_smiles,
                                   result_path=output,
                                   name=f"{args.model_type}_{template_name}",
                                   filter = args.filter)
        sys.stdout.write('Done\n')
        sys.stdout.flush()
        sys.stdout.write('------------------------\n')
        sys.stdout.flush()

if __name__ == '__main__':
    main()