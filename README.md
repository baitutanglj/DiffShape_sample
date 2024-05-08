# DiffShape sample

## Conda environment Dependencies
- cudatoolkit==11.8.0
- hydra-core==1.3.2
- imageio==2.31.1
- numpy==1.25.0
- pandas==2.0.2
- Pillow==9.5.0
- pytorch==2.0.1
- rdkit==2023.03.2
- scikit_learn==1.2.2
- setuptools==68.0.0
- torch_geometric==2.3.1
- torchmetrics==0.11.4
- tqdm==4.65.0

## Download the trained DiffShape model
Download the trained DiffShape model from [Google Cloud Drive](https://drive.google.com/drive/folders/1qTRhD-CvgXCE9cvWX5dHEzDxHsPH6Qck), and then place the downloaded .ckpt file in the ```./outputs/model``` folder

## Sampling example
- sample 1z95_ligand.sdf
```
python DiffShape.py --input ./example/1z95_ligand.sdf --output ./example/outputs/1z95_ligand_everything --batch_size 10 --sample_num 20 --sample_type everything --model_type PointCloudSingle_nstd0.3
```
```
python DiffShape.py --input ./example/1z95_ligand.sdf --output ./example/outputs/1z95_ligand_only_validity --batch_size 10 --sample_num 20 --sample_type only_validity --model_type PointCloudSingle_nstd0.3
```
- geom testset template 10
```
python DiffShape.py --input ./example/geom_testset_H_filter10.sdf --output ./example/outputs/geom10_everything --batch_size 5 --sample_num 10 --sample_type everything --model_type PointCloudSingle_nstd0.3 --filter False
```
```
python DiffShape.py --input ./example/geom_testset_H_filter10.sdf --output ./example/outputs/geom10_only_validity --batch_size 5 --sample_num 10 --sample_type only_validity --model_type PointCloudSingle_nstd0.3 --filter False
```
