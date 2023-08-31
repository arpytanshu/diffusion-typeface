#%%
import torch
import logging

import matplotlib.pyplot as plt

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, model, noise_step=1000, beta_start=0.0001, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_step = noise_step;
        self.beta_start = beta_start;
        self.beta_end = beta_end;
        self.img_size = img_size;
        self.model = model;
        self.device = device;

        self.beta = self.prepare_noise_schedule().to(self.device);
        self.alpha = 1 - self.beta;
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        
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
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)[:, None, None, None]
        eps = torch.randn_like(x0)
        return (sqrt_alpha_hat * x0) + (sqrt_one_minus_alpha_hat * eps)

    def sample(self, n):
        '''
        Notes A:
        Since the amount of noise added in step `t` in the forward process is
        fixed. 
        As per the forward process::
            q(x_t | x_t-1)  = N(x_t; sqrt(1-B_t)*x_t-1, B_t*I)
        Variance of noise added in step `t` is B_t, 
        which is used directly when sampling from p(x_t-1 | xt)
        '''
        logging.info(f"Sampling {n} new images. . .")
        self.model.eval()
        with torch.no_grad():
            


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_step, size=(n,))
    
    


cfg = {
    'noise_step': 100,
    'beta_start': 1e-4,
    'beta_end': 0.02,
    'img_size': 64,
    'device': 'cuda',
}

d = Diffusion(**cfg)
self = d






# %%
