# DiffShape sample

## Conda environment Dependencies
 - dash >= 2.11.1
 - dash-bio >=1.0.2
 - dash-bootstrap-components >= 1.4.1
 - dash-core-components >= 2.0.0
 - dash-cytoscape >= 0.3.0
 - dash-html-components >=2.0.0
 - feffery-antd-components >= 0.2.8
 - rdkit >= 2021.09.1
 - waitress >=2.1.2
 - flask >=2.1.3

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
