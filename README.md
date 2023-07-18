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

The data needed for this project can be obtained from the Smart-TURB portal. Follow these steps to download the data:

1. Visit the [Smart-TURB portal](http://smart-turb.roma2.infn.it).
2. Navigate to `TURB-Lagr` under the `Datasets` section.
3. Click on `Files` -> `data` -> `Lagr_u3c_diffusion.h5`.

Alternatively, you can directly download the data file from this [link](https://smart-turb.roma2.infn.it/init/files/api_file_download/1/___FOLDERSEPARATOR___scratch___FOLDERSEPARATOR___smartturb___FOLDERSEPARATOR___tov___FOLDERSEPARATOR___turb-lagr___FOLDERSEPARATOR___data___FOLDERSEPARATOR___Lagr_u3c_diffusion___POINT___h5/15728642096).

### Data Details and Example Usage

Here is an example of how you can read the data:

```python
import h5py
import numpy as np

with h5py.File('Lagr_u3c_diffusion.h5', 'r') as h5f:
    rx0 = np.array(h5f.get('min'))
    rx1 = np.array(h5f.get('max'))
    u3c = np.array(h5f.get('train'))

velocities = (u3c+1)*(rx1-rx0)/2 + rx0
```

The `u3c` variable is a 3D array with the shape `(327680, 2000, 3)`, representing 327,680 trajectories, each of size 2000, for 3 velocity components. Each component is normalized to the range `[-1, 1]` using the min-max method. The `rx0` and `rx1` variables store the minimum and maximum values for each of the 3 components, respectively. The last line of the code sample retrieves the original velocities from the normalized data.

The data file `Lagr_u3c_diffusion.h5` mentioned above is used for training the `DM-3c` model. For training `DM-1c`, we do not distinguish between the 3 velocity components, thereby tripling the number of trajectories. You can generate the appropriate data by using the [`datasets/preprocessing-lagr_u1c-diffusion.py`](https://github.com/SmartTURB/diffusion-lagr/blob/master/datasets/preprocessing-lagr_u1c-diffusion.py) script. This script concatenates the three velocity components, applies min-max normalization, and saves the result as `Lagr_u1c_diffusion.h5`.

## Training

To train your model, you'll first need to determine certain hyperparameters. We can categorize these hyperparameters into three groups: model architecture, diffusion process, and training flags. Detailed information about these can be found in the [parent repository](https://github.com/openai/improved-diffusion). The run flags for the two models featured in our paper are as follows (please refer to Fig.2 in [the paper](https://arxiv.org/abs/2307.08529)):

For the `DM-1c` model, use the following flags:

```sh
DATA_FLAGS="--dataset_path datasets/Lagr_u1c_diffusion.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 1 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
```

For the `DM-3c` model, you only need to modify `--dataset_path` to `../datasets/Lagr_u3c_diffusion.h5` and `--in_channels` to `3`:

```sh
DATA_FLAGS="--dataset_path datasets/Lagr_u3c_diffusion.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 2000 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 250,125 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
```

After defining your hyperparameters, you can initiate an experiment using the following command:

```sh
mpiexec -n $NUM_GPUS python scripts/turb_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

The training process is distributed, and for our model, we set `$NUM_GPUS` to 4. Note that the `--batch_size` flag represents the batch size on each GPU, so the real batch size is `$NUM_GPUS * batch_size = 256`, as reported in the paper (Fig.2).

### Demo

To assist with testing the software installation and understanding the hyperparameters mentioned above, you can use the smaller dataset `datasets/Lag_u1c_diffusion-demo.h5`, which has a shape of (256, 2000, 3). Note that you do not need to install MPI and parallel h5py for this demo. To run the demo, use the following command:
```sh
python scripts/turb_train.py $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

# dsadasda
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
