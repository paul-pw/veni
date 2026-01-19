# VENI

**[Project Page](https://paul-pw.github.io/veni/)**

If you find VENI useful for your work please cite:

```
@misc{walker2026veni,
  author        = {Paul Walker and James A. D. Gardner and Andreea Ardelean and William A. P. Smith and Bernhard Egger},
  title         = {VENI: Variational Encoder for Natural Illumination},
  eprint        = {},
  archivePrefix = {},
  primaryClass  = {},
  year          = {2026},
  month         = {Jan}
}
```
## Installation

We build on top of Nerfstudio. However, since Nerfstudio is still in very active development with fairly large codebase changes still occurring compatibility might be an issue. Pull requests and issues are very welcome.

#### Install

```bash
git clone https://github.com/paul-pw/veni.git --recurse-submodules
conda create --name veni -y python=3.11
conda activate veni
pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
sudo apt install libopenexr-dev
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
cd ..
pip install -e .
pip install numpy==1.26.4
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

<details>
<summary>installing without `apt install` privileges</summary>

```bash
git clone https://github.com/paul-pw/veni.git
conda create --name veni -y python=3.11
conda activate veni
pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -y -c conda-forge gcc=11 gxx=11
ln -s $CONDA_PREFIX/lib/stubs/libcuda.so $CONDA_PREFIX/lib/libcuda.so
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
conda install -y -c conda-forge openexr
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
cd ..
pip install -e .
pip install numpy==1.26.4
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

</details>

#### Troubleshooting

- `-lcuda not found` 
  - Solution: `ln -s {cuda directory}/lib/stubs/libcuda.so {cuda directory}/lib/libcuda.so`

## Train the model:

To train our VENI Model, run:

`ns-train veni`

in the root directory of this repository.

## Platform

The code is confirmed to work on the following platform:

OS: Debian GNU/Linux 13 (trixie) x86_64  
CPU: 12th Gen Intel(R) Core(TM) i7-12700  
GPU: NVIDIA GeForce RTX 3070  

