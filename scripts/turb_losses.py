"""
Evaluate the losses for a diffusion model.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.turb_datasets import load_data
from guided_diffusion.script_util import (
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
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    import os
    seed = 0*4 + int(os.environ["CUDA_VISIBLE_DEVICES"])
    th.manual_seed(seed)
    run_losses_evaluation(model, diffusion, data, args.num_samples)


def run_losses_evaluation(model, diffusion, data, num_samples):
    all_total_loss = []
    all_losses = {"loss": [], "vb": [], "mse": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_losses = diffusion.calc_losses_loop(
            model, batch, model_kwargs=model_kwargs
        )

        for key in minibatch_losses:
            losses = minibatch_losses[key]
            gathered_losses = [th.zeros_like(losses) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, losses)  # gather not supported with NCCL
            all_losses[key].extend([losses.cpu().numpy() for losses in gathered_losses])

        total_loss = minibatch_losses["loss"]
        total_loss = total_loss.mean() / dist.get_world_size()
        dist.all_reduce(total_loss)
        all_total_loss.append(total_loss.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.log(f"done {num_complete} samples: total_loss={np.mean(all_total_loss)}")

    if dist.get_rank() == 0:
        for name in minibatch_losses:
            losses = np.concatenate(all_losses[name])
            shape_str = "x".join([str(x) for x in losses.shape])
            out_path = os.path.join(logger.get_dir(), f"{name}_losses_{shape_str}.npz")
            logger.log(f"saving {name} losses to {out_path}")
            np.savez(out_path, losses)

    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        dataset_path="", dataset_name="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
