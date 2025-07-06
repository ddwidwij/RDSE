import os
import sys
import torch
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, Unet, UnetRes, set_seed)
from datasets.audio_dataset import SpecsDataModule


if __name__ == '__main__':
    # init
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
    sys.stdout.flush()
    set_seed(10)
    debug = False

    if debug:
        save_and_sample_every = 2
        sampling_timesteps = 10
        sampling_timesteps_original_ddim_ddpm = 10
        train_num_steps = 200
    else:
        save_and_sample_every = 1000
        if len(sys.argv) > 1:
            sampling_timesteps = int(sys.argv[1])
        else:
            sampling_timesteps = 1
        sampling_timesteps_original_ddim_ddpm = 250
        train_num_steps = 300000

    condition = True
    input_condition = False

    if condition:
        # Image restoration
        base_dir = "/home/raoziyu/桌面/data/VoiceBank-DEMAND-16k"
        results_folder = "./results/model"
        Dataset = SpecsDataModule(base_dir=base_dir, format='De-noising', batch_size=4)
        train_batch_size = 4
        num_samples = 1
        sum_scale = 0.01
        spec_size = 256

    num_unet = 1
    objective = 'pred_res'
    test_res_or_noise = "res"
    model = Unet(
        dim=32,
        dim_mults=(1, 1, 1, 1),
        condition=condition,
        input_condition=input_condition,
    )
    diffusion = ResidualDiffusion(
        model,
        spec_size=spec_size,
        timesteps=1000,  # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='mse',  # mse or mea
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        resume=False
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
        stage='fit',
        results_folder=results_folder,
    )
    # restore train
    # trainer.load(10)

    # train
    # torch.autograd.set_detect_anomaly(True)
    trainer.train(results_folder)




