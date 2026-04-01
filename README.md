# VENI

**[Project Page](https://paul-pw.github.io/veni/) | [Paper](https://arxiv.org/abs/2601.14079)**

If you find VENI useful for your work please cite:

```
@misc{walker2026veni,
  title={VENI: Variational Encoder for Natural Illumination}, 
  author={Paul Walker and James A. D. Gardner and Andreea Ardelean and William A. P. Smith and Bernhard Egger},
  year={2026},
  eprint={2601.14079},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2601.14079}, 
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

## Download Data and Pretrained Models

```bash
# Download RENI++ Dataset
python3 ns_reni/scripts/download_data.py ./data/

# (optional) Download RENI++ models for comparison with our models
python3 ns_reni/scripts/download_models.py ./outputs/reni/

# Download VENI Models
wget -O veni_models.zip "https://www.dropbox.com/scl/fi/bekgqob30xjdxb1bjormf/veni_models.zip?rlkey=q6zu6myd0o0zn5gpbd41mskff&st=6oqxw1xq&dl=1"
unzip veni_models.zip -d ./outputs/veni/
```

## Train the model:

To train our VENI Model, run:

`ns-train veni`

in the root directory of this repository.

## Platform

The code is confirmed to work on the following platform:

OS: Debian GNU/Linux 13 (trixie) x86_64  
CPU: 12th Gen Intel(R) Core(TM) i7-12700  
GPU: NVIDIA GeForce RTX 3070  

