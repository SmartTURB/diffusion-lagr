# diffusion-lagr

This is the codebase for [Synthetic Lagrangian Turbulence by Generative Diffusion Models](https://arxiv.org/abs/2307.08529).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications specifically tailored to adapt the Lagrangian turbulence data in the Smart-TURB portal http://smart-turb.roma2.infn.it, under the [TURB-Lagr](https://smart-turb.roma2.infn.it/init/routes/#/logging/view_dataset/2/tabmeta) dataset.

# Usage

## Development Environment

Our software was developed and tested on a system with the following specifications:

- **Operating System**: Ubuntu 20.04.4 LTS
- **Python Version**: 3.7.16
- **PyTorch Version**: 1.13.1
- **MPI Implementation**: OpenRTE 4.0.2
- **CUDA Version**: 11.5
- **GPU Model**: NVIDIA A100

## Installation

We recommend using a Conda environment to manage dependencies. The code relies on the MPI library and [parallel h5py](https://docs.h5py.org/en/stable/mpi.html). After setting up your environment, clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `guided_diffusion` python package that the scripts depend on.

### Troubleshooting Installation

During the installation process, you might encounter a couple of known issues. Here are some tips to help you resolve them:

1. **Parallel h5py Installation**: The installation of parallel h5py can be problematic. To bypass this issue, you can modify the code at [this line](https://github.com/SmartTURB/diffusion-lagr/blob/master/guided_diffusion/turb_datasets.py#L75).
2. **PyTorch Installation**: In our experience, sometimes it's necessary to reinstall PyTorch depending on your system environment. You can download and install PyTorch from their [official website](https://pytorch.org/).

## Preparing Data

The training code reads images from a directory of image files. In the [datasets](datasets) folder, we have provided instructions/scripts for preparing these directories for ImageNet, LSUN bedrooms, and CIFAR-10.

For creating your own dataset, simply dump all of your images into a directory with ".jpg", ".jpeg", or ".png" extensions. If you wish to train a class-conditional model, name the files like "mylabel1_XXX.jpg", "mylabel2_YYY.jpg", etc., so that the data loader knows that "mylabel1" and "mylabel2" are the labels. Subdirectories will automatically be enumerated as well, so the images can be organized into a recursive structure (although the directory names will be ignored, and the underscore prefixes are used as names).

The images will automatically be scaled and center-cropped by the data-loading pipeline. Simply pass `--data_dir path/to/images` to the training script, and it will take care of the rest.

# Training models and Sampling

Training diffusion modelss and Sampling are described in the [parent repository](https://github.com/openai/improved-diffusion).

For DM-1c training:

```sh
DATA_FLAGS="--dataset_path /mnt/petaStor/li/Job/diffusion-lagr/datasets/Lagr_u1c_diffusion.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 1 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
mpiexec -n 4 python ../scripts/turb_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

and sampling:

```sh
SAMPLE_FLAGS="--num_samples 179200 --batch_size 64 --model_path /mnt/petaStor/li/Job/diffusion-lagr/lagr_u1c-IS2000-NC128-NRB3-DS800-NStanh6_1-LR1e-4-BS256-train/ema_0.9999_250000.pt"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 1 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
mpiexec -n 4 python ../scripts/turb_sample.py $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```

For DM-3c training:

```sh
DATA_FLAGS="--dataset_path /mnt/petaStor/li/Job/diffusion-lagr/datasets/Lagr_u3c_diffusion.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
mpiexec -n 4 python ../scripts/turb_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

and sampling:

```sh
SAMPLE_FLAGS="--num_samples 179200 --batch_size 64 --model_path /mnt/petaStor/li/Job/diffusion-lagr/lagr_u3c-IS2000-NC128-NRB3-DS800-NStanh6_1-LR1e-4-BS256-train/ema_0.9999_400000.pt"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
mpiexec -n 4 python ../scripts/turb_sample.py $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
