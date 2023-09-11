
import os
import logging
import numpy as np
from time import time
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch import nn

from utils import Config
from modules import Unet, EMA
from diffusion import Diffusion
from default_config import config
from utils import setup_logging, get_data, save_images, progress_bar


def train(args):
    setup_logging(args)
    dataloader = get_data(args)
    model = Unet(device=args.device, num_classes=args.num_classes)
    start_epoch = 1

    if os.path.exists(os.path.join(args.run_name, 'models', 'ckpt.pt')):
        chkpt = torch.load(os.path.join(args.run_name, 'models', 'ckpt.pt'))
        model.load_state_dict(chkpt['model'])
        start_epoch = chkpt['epoch']

    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=args.device)

    l = len(dataloader)
    ema = EMA(beta=args.ema_beta)
    ema_model = deepcopy(model).eval().requires_grad_(False)

    for epoch in range(start_epoch, args.epochs+1):
        logging.info('Starting epoch {epoch}:')
        tick = time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            t = diffusion.sample_timesteps(images.shape[0]).to(torch.long).to(args.device)
            x_t, noise = diffusion.noise_image(images, t)
            if np.random.random() < args.perc_uncond_train:
                labels = None
            predicted_noise = model(x_t, t, labels)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            progress_bar(i, len(dataloader), 50, f"epoch:{epoch} elapsed:{time()-tick:.3f}")
        
        print(f"epoch:{epoch}, MSE:{loss.item()}")
        
        
        if (epoch%args.log_interval) == 0:
            labels = torch.arange(args.log_batch).long().to(args.device)
            sampled_images = diffusion.sample_cond(model, args.log_batch, labels)

            save_images(sampled_images, os.path.join(args.run_name, "results", f"{epoch}.jpg"))

            chkpt = {'model':model.state_dict(), 'ema_model': ema_model.state_dict(), 'epoch':epoch}
            torch.save(chkpt, os.path.join(args.run_name, "models", f"ckpt.pt"))


args = Config(config=config)
train(args)
