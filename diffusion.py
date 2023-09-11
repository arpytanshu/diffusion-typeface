
import torch
import logging
from utils import progress_bar
from time import time
from copy import deepcopy
import numpy as np

class Diffusion:
    def __init__(self, noise_step=1000, beta_start=0.0001, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_step = noise_step;
        self.beta_start = beta_start;
        self.beta_end = beta_end;
        self.img_size = img_size;
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
    

    def sample(self, model, n, labels, cfg_scale=3, step=1, return_intermediate=False):
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
        if return_intermediate:
            intermediate_images = []

        model.eval()
        with torch.no_grad():
            # make some pure gaussian noise to begin with
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            tick = time()
            for i in reversed(range(1, self.noise_step)[::step]):
                t = (torch.ones(n) * i).long().to(self.device)
                beta = self.beta[t][:, None, None, None]
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]

                noise = torch.randn_like(x) if (i > 1) else torch.zeros_like(x)

                eps_theta = model(x, t, labels) # predicted_noise
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    eps_theta = torch.lerp(uncond_predicted_noise, eps_theta, cfg_scale)

                x = 1 / torch.sqrt(alpha) * (x - ( (1-alpha) / (torch.sqrt(1-alpha_hat))) * eps_theta) + torch.sqrt(beta) * noise
                
                if x.isnan().sum().item() > 0:
                    print(f"x became all Nans at timestep:{i}")
                    break;
                
                if return_intermediate:
                    x_ =  deepcopy(x).cpu()
                    x_ = (x_.clamp(-1, 1) + 1) / 2
                    x_ = (x_ * 255).type(torch.uint8)
                    intermediate_images.append(x_)
                
                progress_bar(self.noise_step - i, self.noise_step, 50, f"Diffusing... elapsed:{time()-tick:.3f}" )

        model.train()

        if return_intermediate:
            return intermediate_images
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_step, size=(n,))
    
