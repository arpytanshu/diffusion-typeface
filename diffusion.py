#%%

import torch
import logging

import matplotlib.pyplot as plt
from time import time
from utils import setup_logging, get_data, plot_images, save_images, progress_bar
from modules import Unet
from torch.optim import AdamW
from torch import nn
import os

# logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, model, noise_step=1000, beta_start=0.0001, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_step = noise_step;
        self.beta_start = beta_start;
        self.beta_end = beta_end;
        self.img_size = img_size;
        self.model = model;
        self.device = device;

        self.beta = self.prepare_noise_schedule().to(self.device);
        self.alpha = (1 - self.beta)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0);

        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_step);

    def noise_image(self, x0, t):
        '''
        add noise to x0 - OR - sample from q(x_t | x_0)

        Forward diffusion::
            q(x_t | x_t-1)  = N(x_t; sqrt(1-B_t)*x_t-1, B_t*I)
            - or - after reparameterizations -
            q(x_t | x_0)    = N(x_t; sqrt(alpha_hat)*x_0, (1 - alpha_hat)*I)
            
        OR, simply.
            x_t = sqrt(alpha_hat)*x0 + sqrt(1 - alpha_hat)*eps
            i.e. keeping sqrt(alpha_hat) ratio of x0 and sqrt(1-alpha_hat) ratio of gaussian noise.

        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x0)
        return (sqrt_alpha_hat * x0) + (sqrt_one_minus_alpha_hat * eps), eps

    def sample(self, n):
        '''
        Implements: "Algorithm 2: Sampling
        Sample from p_theta(x_t-1 | x_t)

        Notes A:
        Since the amount of noise added in step `t` in the forward process is
        fixed, as per the forward process::
            q(x_t | x_t-1)  = N(x_t; sqrt(1-B_t)*x_t-1, B_t*I)
        
        Variance of noise added in step `t` is B_t, 
        which is used directly when sampling from p(x_t-1 | xt)
        '''
        logging.info(f"Sampling {n} new images. . .")
        self.model.eval()
        with torch.no_grad():
            # make some pure gaussian noise to begin with
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_step)):
                t = (torch.ones(n) * i).long().to(self.device)
                beta = self.beta[t][:, None, None, None]
                alpha = self.alpha[t][:, None, None, None]                
                alpha_hat = self.alpha_hat[t][:, None, None, None]

                noise = torch.randn_like(x) if (i > 1) else torch.zeros_like(x)

                eps_theta = self.model(x, t).to(torch.float32) # predicted_noise
                # eps_theta_coeff = beta / torch.sqrt(1-alpha_hat)
                # x = (1 / torch.sqrt(alpha_hat)) * (x - (eps_theta_coeff * eps_theta)) + (torch.sqrt(beta) * noise)

                x = 1 / torch.sqrt(alpha) * (x - ( (1-alpha) / (torch.sqrt(1-alpha_hat))) * eps_theta) + torch.sqrt(beta) * noise
                if x.isnan().sum().item() > 0:
                    print(f"x became all Nans at timestep:{i}")
                    break;

        self.model.train()

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_step, size=(n,))
    


def train(args):
    setup_logging(args)
    device = args.device
    dataloader = get_data(args)
    model = Unet(device=args.device)
    start_epoch = 1

    if os.path.exists(os.path.join(args.run_name, 'models', 'ckpt.pt')):
        chkpt = torch.load(os.path.join(args.run_name, 'models', 'ckpt.pt'))
        model.load_state_dict(chkpt['model'])
        start_epoch = chkpt['epoch']

    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(model=model, img_size=args.img_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(dataloader)

    for epoch in range(start_epoch, args.epochs+1):
        logging.info('Starting epoch {epoch}:')
        tick = time()
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(torch.long).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # torch.cuda.empty_cache()

            # logger.add_scalar("MSE", loss.item(), )
            progress_bar(i, len(dataloader), 50, f"epoch:{epoch} elapsed:{time()-tick:.3f}")
        
        print(f"epoch:{epoch}, MSE:{loss.item()}")
        
        
        if (epoch%5) == 0:
            sampled_images = diffusion.sample(n=8)
            save_images(sampled_images, os.path.join(args.run_name, "results", f"{epoch}.jpg"))
            chkpt = {'model':model.state_dict(), 'epoch':epoch}
            torch.save(chkpt, os.path.join(args.run_name, "models", f"ckpt.pt"))


def load_model(path, args):
    model = Unet(device=args.device)
    chkpt = torch.load(path)
    

class Config():
    def __init__(self, d):
        self.__dict__ = d

cfg = {
    "run_name" : "DDPM_unconditional2",
    "epochs" : 500,
    "batch_size" : 16,
    "img_size" : 64,
    "dataset_path" : "/shared/datasets/LandscapeDataset-kaggle/",
    "device" : "cuda",
    "lr" : 3e-4,
}
args = Config(cfg)
train(args)





# %%
