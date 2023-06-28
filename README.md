# diffusion-lagr

This is the codebase for [Synthetic Lagrangian Turbulence by Generative Diffusion Models](TODO)

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications specifically tailored to adapt the Lagrangian turbulence data in the Smart-TURB portal http://smart-turb.roma2.infn.it, under the TURB-Lagr repository.

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
