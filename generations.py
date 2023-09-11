

#%%

import os
import numpy as np
from utils import plot_images, save_images, progress_bar
import matplotlib.pyplot as  plt

from time import time
import torch
from modules import Unet
from diffusion import Diffusion


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

def get_labels(string):
    chars = "abcdefghijklmnopqrstuvwxyz"
    ix_char_map = {ix:k for ix, k in enumerate(chars)}
    char_ix_map = {k:ix for ix, k in enumerate(chars)}
    acceptable_chars = lambda string : [ch for ch in string.lower() if ch in chars]
    map_chars_to_ix = lambda string : [char_ix_map[ch] for ch in string]
    string = acceptable_chars(string)
    labels = map_chars_to_ix(string)
    return labels

args = Config(cfg)
device = args.device


chkpt = torch.load(os.path.join(args.run_name, 'models', 'ckpt.pt'))
model = Unet(device=args.device, num_classes=args.num_classes)
model.load_state_dict(chkpt['model'])

diffusion = Diffusion(img_size=args.img_size, device=device)
model.eval()



#%% showcase generations

string = 'demonstration'
labels = get_labels(string)
labels = torch.tensor(labels).long().to(device)


for ix in range(20, 40):
    cfg_scale = np.random.randint(3, 10)
    sampled_images = diffusion.sample(model, len(labels), labels=labels, cfg_scale=cfg_scale)
    sampled_images = torch.cat([i for i in sampled_images.cpu()], dim=-1).permute(1,2,0)
    fig = plt.figure(figsize=(10, 2))
    plt.imshow(sampled_images)
    plt.xticks([]); plt.yticks([]); plt.box(False)
    fig.tight_layout()
    plt.savefig(f"assets/{string}_{ix}.png")
    plt.close()
    

#%% showcase diffusion process

string = 'diffusion'
labels = get_labels(string)
labels = torch.tensor(labels).long().to(device)
sampled_images = diffusion.sample(model, len(labels), labels=labels, cfg_scale=5, return_intermediate=True)

ix = np.arange(0, 200, 9)
ix = np.hstack([ix, np.arange(200, 400, 8)])
ix = np.hstack([ix, np.arange(400, 600, 6)])
ix = np.hstack([ix, np.arange(600, 800, 4)])
ix = np.hstack([ix, np.arange(800, 999, 3)])
print(ix)

sampled_images_ixed = torch.stack(sampled_images)[ix]
# torch.flip(sampled_images, dims=[0])

for ix, img in enumerate(sampled_images_ixed):
    img = torch.cat([i for i in img.cpu()], dim=-1).permute(1,2,0)
    
    fig = plt.figure(figsize=(5, 1))
    plt.imshow(img)
    plt.xticks([]); plt.yticks([]); plt.box(False)
    fig.tight_layout()
    plt.savefig(f"DDPM_cond1/results2/diffusion_{ix}.png")
    plt.close()


# %%
