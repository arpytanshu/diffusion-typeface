

#%%

import os
import numpy as np
from utils import setup_logging, get_data, plot_images, save_images, progress_bar


from time import time
import torch
from modules import Unet, EMA
from torch.optim import AdamW
from torch import nn
from diffusion import Diffusion
import logging
from copy import deepcopy


def train(args):
    setup_logging(args)
    device = args.device
    dataloader = get_data(args)
    model = Unet(device=args.device, num_classes=args.num_classes)
    start_epoch = 1

    if os.path.exists(os.path.join(args.run_name, 'models', 'ckpt.pt')):
        chkpt = torch.load(os.path.join(args.run_name, 'models', 'ckpt.pt'))
        model.load_state_dict(chkpt['model'])
        start_epoch = chkpt['epoch']

    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(dataloader)
    ema = EMA(beta=0.99)
    ema_model = deepcopy(model).eval().requires_grad_(False)

    for epoch in range(start_epoch, args.epochs+1):
        logging.info('Starting epoch {epoch}:')
        tick = time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(torch.long).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)


            # logger.add_scalar("MSE", loss.item(), )
            progress_bar(i, len(dataloader), 50, f"epoch:{epoch} elapsed:{time()-tick:.3f}")
        
        print(f"epoch:{epoch}, MSE:{loss.item()}")
        
        
        if (epoch%7) == 0:
            labels = torch.arange(16).long().to(device)
            ema_sampled_images = diffusion.sample_cond(ema_model, 16, labels)
            sampled_images = diffusion.sample_cond(model, 16, labels)

            save_images(sampled_images, os.path.join(args.run_name, "results", f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join(args.run_name, "results", f"ema_{epoch}.jpg"))

            chkpt = {'model':model.state_dict(), 'ema_model': ema_model.state_dict(), 'epoch':epoch}
            torch.save(chkpt, os.path.join(args.run_name, "models", f"ckpt.pt"))


def load_model(path, args):
    model = Unet(device=args.device)
    chkpt = torch.load(path)
    

class Config():
    def __init__(self, d):
        self.__dict__ = d

cfg = {
    "run_name" : "DDPM_cond1",
    "epochs" : 500,
    "batch_size" : 16,
    "img_size" : 64,
    "dataset_path" : "/shared/datasets/english_typeface_classes/",
    "device" : "cuda",
    "lr" : 3e-4,
    "num_classes": 26,
}
args = Config(cfg)
train(args)



# %%
