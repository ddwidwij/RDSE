import os
import sys
import argparse
from datasets.audio_dataset import SpecsDataModule
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, Unet, UnetRes, set_seed)


def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/mnt/Datasets/Restoration')
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size') #568
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', type=bool, default=True, help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--bsize", type=int, default=2)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    # init
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
    sys.stdout.flush()
    set_seed(10)

    save_and_sample_every = 1000
    if len(sys.argv) > 1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 5
    train_num_steps = 100000

    condition = True
    input_condition = False
    input_condition_mask = False

    train_batch_size = 1
    num_samples = 1
    spec_size = 256

    opt = parsr_args()

    results_folder = "/home/rzy/code/RDDM-TS/results/pre_model"
    base_dir = "/home/rzy/data/VoiceBank-DEMAND-16k"
    format = 'De-noising'

    Dataset = SpecsDataModule(base_dir=base_dir, format=format, batch_size=1)

    num_unet = 1
    objective = 'pred_res'
    test_res_or_noise = "res"
    sampling_timesteps = 1
    sum_scale = 0.01
    ddim_sampling_eta = 0.

    model = UnetRes(
        dim=32,
        dim_mults=(1, 1, 1, 1),
        num_unet=num_unet,
        condition=condition,
        input_condition=input_condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise
    )
    diffusion = ResidualDiffusion(
        model,
        spec_size=spec_size,
        timesteps=1000,  # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='mse',  # mse or mae
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        use_mean_value=True,
        midpoint_time=-1,
    )

    trainer = Trainer(
        diffusion,
        Dataset=Dataset,
        train_batch_size=train_batch_size,
        num_samples=num_samples,
        train_lr=8e-5,
        train_num_steps=train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        condition=condition,
        save_and_sample_every=save_and_sample_every,
        crop_patch=False,
        num_unet=num_unet,
        stage='test',
        results_folder=results_folder
    )

    # test
    if not trainer.accelerator.is_local_main_process:
        pass
    else:
        trainer.load(280)
        trainer.set_results_folder('/home/rzy/results')
        trainer.test(last=True)
