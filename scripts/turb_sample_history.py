"""
Generate a large batch of Lagrangian trajectories from a model and save them as a large
numpy array. This can be used to produce samples for statistical evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    #noise = th.zeros(
    # noise = th.ones(
    #     (args.batch_size, args.in_channels, args.image_size),
    #     dtype=th.float32,
    #     device=dist_util.dev()
    # ) * 2
    # noise = th.from_numpy(
    #     np.load('../velocity_module-IS64-NC128-NRB3-DS4000-NScosine-LR1e-4-BS256-sample/fixed_noise_64x1x64x64.npy')
    # ).to(dtype=th.float32, device=dist_util.dev())
    import os
    seed = 0*8 + int(os.environ["CUDA_VISIBLE_DEVICES"])
    th.manual_seed(seed)
    curr_batch, num_complete = 0, 0
    while num_complete < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = diffusion.p_sample_loop_history
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size),
            #noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample[:, -1] = sample[:, -1].clamp(-1, 1)
        sample = sample.permute(0, 1, 3, 2)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images = [sample.cpu().numpy() for sample in gathered_samples]
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels = [labels.cpu().numpy() for labels in gathered_labels]
        curr_batch += 1
        num_complete += dist.get_world_size() * args.batch_size
        logger.log(f"created {num_complete} samples")

        arr = np.concatenate(all_images, axis=0)
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(
                logger.get_dir(), "samples_history-seed0",
                f"batch{curr_batch:03d}-samples_history_{shape_str}.npz"
            )
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
