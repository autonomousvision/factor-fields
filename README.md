# Factor Fields
## [Project page](https://apchenstu.github.io/FactorFields/) |  [Paper](https://arxiv.org/abs/2302.01226)
This repository contains a pytorch implementation for the paper: [Factor Fields: A Unified Framework for Neural Fields and Beyond](https://arxiv.org/abs/2302.01226) and [Dictionary Fields: Learning a Neural Basis Decomposition](https://arxiv.org/abs/2302.01226). Our work present a novel framework for modeling and representing signals, 
we have also observed that Dictionary Fields offer benefits such as improved **approximation quality**, **compactness**, **faster training speed**, and the ability to **generalize** to unseen images and 3D scenes.<br><br>


## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.13.0 

Install environment:
```sh
conda create -n FactorFields python=3.9
conda activate FactorFields
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt 
```

Optionally install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), only needed if you want to run hash grid based representations.
```sh
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```


# Quick Start
Please ensure that you download the corresponding dataset and extract its contents into the `data` folder.

## Image
* [Data - Image Set](https://1drv.ms/u/c/0c624178fab774b7/Ebd0t_p4QWIggAx3BAAAAAABikvhj5m_rVm1-qIpYFyrFg?e=hyTeZf)

The training script can be found at `scripts/2D_regression.ipynb`, and the configuration file is located at `configs/image.yaml`.

<p align="left">
  <img src="media/Girl_with_a_Pearl_Earring.jpg" alt="Girl with a Pearl Earring" width="320">
</p>

## SDF
* [Data - Mesh set](https://1drv.ms/u/c/0c624178fab774b7/Ebd0t_p4QWIggAx4BAAAAAABbouT0SD3PCChlfTQJL3XzA?e=ImcsAj)

The training script can be found at `scripts/sdf_regression.ipynb`, and the configuration file is located at `configs/sdf.yaml`.

<img src="https://github.com/apchenstu/GIFs/blob/main/FactorField-statuette.gif" alt="GIF" width="500px">



## NeRF
* [Data - Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Data-Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)

The training script can be found at `train_per_scene.py`:

```python
python train_per_scene.py configs/nerf.yaml defaults.expname=lego dataset.datadir=./data/nerf_synthetic/lego
```

<img src="https://github.com/apchenstu/GIFs/blob/main/FactorField-mic.gif" alt="GIF" width="500px"


## Generalization Image
* [Data - FFHQ](https://github.com/NVlabs/ffhq-dataset)

The training script can be found at `2D_set_regression.ipynb`

<p align="left">
  <img src="media/inpainting.png" alt="Inpainting" width="640">
</p>



## Generalization NeRF
* [Data - Google Scanned Objects](https://drive.google.com/file/d/1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2/view)

```python
python train_across_scene.py configs/nerf_set.yaml
```

<img src="https://github.com/apchenstu/GIFs/blob/main/FactorField-few-shot.gif" alt="GIF" width="500px">


## More examples

Command explanation with a nerf example:
* `model.basis_dims=[4, 4, 4, 2, 2, 2]` adjusts the number of levels and channels at each level, with a total of 6 levels and 18 channels.
* `model.basis_resos=[32, 51, 70, 89, 108, 128]` represents the resolution of the feature embeddings.
* `model.freq_bands=[2.0, 3.2, 4.4, 5.6, 6.8, 8.0]` indicates the frequency parameters applied at each level of the coordinate transformation function.
* `model.coeff_type` represents the coefficient field representations and can be one of the following: [none, x, grid, mlp, vec, cp, vm].
* `model.basis_type` represents the basis field representation and can be one of the following: [none, x, grid, mlp, vec, cp, vm, hash].
* `model.basis_mapping` represents the coordinate transformation and can be one of the following: [x, triangle, sawtooth, trigonometric]. Please note that if you want to use orthogonal projection, choose the cp or vm basis type, as they automatically utilize the orthogonal projection functions.
* `model.total_params` controls the total model size. It is important to note that the model's size capability is determined by model.basis_resos and model.basis_dims. The total_params parameter mainly affects the capability of the coefficients.
* `exportation.render_only` you can rendering item after training by setting this label to 1. Please also specify the `defaults.ckpt` label.
* `exportation....` you can specify whether to render the items of `[render_test, render_train, render_path, export_mesh]` after training by enable the corressponding label to 1.

Some pre-defined configurations (such as occNet, DVGO, nerf, iNGP, EG3D) can be found in `README_FactorField.py`.


## COPY RIGHT
* [Summer Day](https://www.rijksmuseum.nl/en/collection/SK-A-3005) - Credit goes to Johan Hendrik Weissenbruch and rijksmuseum.
* [Mars](https://solarsystem.nasa.gov/resources/933/true-colors-of-pluto/) - Credit goes to NASA.
* [Albert](https://cdn.loc.gov/service/pnp/cph/3b40000/3b46000/3b46000/3b46036v.jpg) - Credit goes to Orren Jack Turner.
* [Girl With a Pearl Earring](http://profoundism.com/free_licenses.html) - Renovation copyright Koorosh Orooj (CC BY-SA 4.0).


## Citation
If you find our code or paper helpful, please consider citing both of these papers:
```
@article{Chen2023factor,
  title={Factor Fields: A Unified Framework for Neural Fields and Beyond},
  author={Chen, Anpei and Xu, Zexiang and Wei, Xinyue and Tang, Siyu and Su, Hao and Geiger, Andreas},
  journal={arXiv preprint arXiv:2302.01226},
  year={2023}
}

@article{Chen2023SIGGRAPH, 
 title={{Dictionary Fields: Learning a Neural Basis Decomposition}}, 
 author={Anpei, Chen and Zexiang, Xu and Xinyue, Wei and Siyu, Tang and Hao, Su and Andreas, Geiger}, 
 booktitle={International Conference on Computer Graphics and Interactive Techniques (SIGGRAPH)}, 
 year={2023}}
```
