# GaussianHead: High-fidelity Head Avatars with Learnable Gaussian Derivation
### | [arXiv](https://arxiv.org/pdf/2312.01632.pdf) | [Project Page](https://chiehwangs.github.io/gaussian-head-page/) |
![](assets/teaser-v2.png)



## Getting Started
* Git clone this repo, note using `--recursive` to get submodules;
* Create a conda or python environment and activate. For e.g.,`conda create -n gaussian-head python=3.8`, `source(or conda) activate gaussian-head`;
* [PyTorch](https://pytorch.org/get-started/previous-versions/) >= 2.0.0 is necessary as geoopt requires, for e.g., `pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`;
* install all requirements in `requirements.txt`;
* [geoopt](https://github.com/geoopt/geoopt) is necessary for Riemannian ADAM, refer to it and install in pypi by `pip install geoopt`.

## Riemannian ADAM
Please refer to [here](https://github.com/geoopt/geoopt) to download it, and please consider citing '*Riemannian Adaptive Optimization Methods*' in ICLR2019 if used.

## Preparing Dataset
All our data is sourced from publicly available datasets [NeRFBlendShape](https://drive.google.com/drive/folders/1OiUvo7vHekVpy67Nuxnh3EuJQo7hlSq1?usp=sharing) and make specific modifications. 

Download [our modified datasets](https://drive.google.com/file/d/1vriFnMGsXPVTWRsHQ37SmMNZxU17nICT/view?usp=sharing) for train and render, store it in the following directory.

```
gaussian-head
    ├── data
       ├── id1
           ├── ori_imgs    # rgb frames
           ├── mask    # binary masks
           └── transforms.json    # camera params and expressions
       ├── id2
           ......
```

## Pre-trained Model
Download the [id1 pre-trained model](https://drive.google.com/file/d/13SjlhQ7MOONPUenJHbqwdGJoGeU2Arz6/view?usp=sharing) (training on RTX 2080ti) to quickly view the results, and store the training model according to `./gaussian-head/output/id1`

## Training[soon...]
Store the training data according to the format and cd to `./gaussian-head`, run:
```
python ./train.py -s ./data/${id} -m ./output/${id} --eval
```

## Rendering
Use your own trained model or the pre-trained model we provide, cd to `./gaussian-head` and run next command, output results will save in `./gaussian-head/output/id1/test`
```
python render.py -m ./output/${id}
```

## Additional Tools
>- Set `--is_debug` used to quickly load a small amount of training data for debug;
>- After training, set `--novel_view`, and then run  `render.py` to get the novel perspective result rotated by the y-axis;
>- Set `--only_head` will only perform head training and rendering. Before this, face_parsing needs to be performed to obtain the segmentation, this can be easily obtained at [NeRFBlendShape](https://drive.google.com/drive/folders/1OiUvo7vHekVpy67Nuxnh3EuJQo7hlSq1?usp=sharing);

## Citation
If anything useful, a star is best and please cite as:
```
@misc{wang2024gaussianhead,
      title={GaussianHead: High-fidelity Head Avatars with Learnable Gaussian Derivation}, 
      author={Jie Wang and Jiu-Cheng Xie and Xianyan Li and Feng Xu and Chi-Man Pun and Hao Gao},
      year={2024},
      eprint={2312.01632},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
