import os
import sys
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import sys
import numpy as np


def plot_images(images):
    plt.figure(figsize=(20, 20))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path,  transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader 

def setup_logging(args):    
    os.makedirs(args.run_name, exist_ok=True)
    os.makedirs(os.path.join(args.run_name, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.run_name, "results"), exist_ok=True)


def progress_bar(current, total, bar_length=50, text="Progress"):
    percent = float(current) / total
    abs = f"{{{current} / {total}}}"
    arrow = '|' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}% {3}".format(text, arrow + spaces, int(round(percent * 100)), abs))
    sys.stdout.flush()


def get_labels(string):
    chars = "abcdefghijklmnopqrstuvwxyz"
    char_ix_map = {k:ix for ix, k in enumerate(chars)}
    acceptable_chars = lambda string : [ch for ch in string.lower() if ch in chars]
    map_chars_to_ix = lambda string : [char_ix_map[ch] for ch in string]
    string = acceptable_chars(string)
    labels = map_chars_to_ix(string)
    return labels


class Config:
    def __init__(self, **kwargs):
        path = kwargs.get('path')
        config = kwargs.get('config')
        if path is not None:
            self.load_json(path)
        elif config is not None:
            self.load_state_dict(config)
        else:
            raise Exception("At least one of 'config' or 'path' must be provided.")
    
    def state_dict(self): # get the config values as a dictionary.
        dump_dict = {}
        for k in self.__dict__.keys():
            v = self.__dict__[k]
            if type(v) == type(self):
                dump_dict[k] = v.__dict__
            else:
                dump_dict[k] = v
        return dump_dict

    def save_json(self, path): # save a Config object into a json file.
        dump_dict = self.state_dict()
        with open(path, 'w') as f:
            json.dump(dump_dict, f)
    
    def load_json(self, path): # load a json file into Config object.
        with open(path, 'r') as f:
            load_dict = json.load(f)
        self.load_state_dict(load_dict)
    
    def load_state_dict(self, d): # load a dictionary into Config object.
        for k in d.keys():
            v = d[k]
            if type(v) == dict:
                self.__dict__[k] = Config(config=v)
            else:
                self.__dict__[k] = v

    def __repr__(self):
        s = []
        for k in self.__dict__.keys():
            v = self.__dict__[k]
            if type(v) == type(self): # this is top level config.
                s.append(k)
                s.append(self.__dict__[k].__repr__())
            else: # this is last level config.
                s.append('\t' + k +':\t'+ str(v))
        return '\n'.join(s)

def plot_diffusion_steps(args, dst_path, string, model, diffusion):
    string = 'diffusion'
    labels = get_labels(string)
    labels = torch.tensor(labels).long().to(args.device)
    sampled_images = diffusion.sample(model, len(labels), labels=labels, cfg_scale=args.cfg_scale, return_intermediate=True)

    ix = np.arange(0, 200, 9)
    ix = np.hstack([ix, np.arange(200, 400, 8)])
    ix = np.hstack([ix, np.arange(400, 600, 6)])
    ix = np.hstack([ix, np.arange(600, 800, 4)])
    ix = np.hstack([ix, np.arange(800, 999, 3)])
    print(ix)

    sampled_images_ixed = torch.stack(sampled_images)[ix]
    for ix, img in enumerate(sampled_images_ixed):
        img = torch.cat([i for i in img.cpu()], dim=-1).permute(1,2,0)
        
        fig = plt.figure(figsize=(5, 1))
        plt.imshow(img)
        plt.xticks([]); plt.yticks([]); plt.box(False)
        fig.tight_layout()
        plt.savefig(os.path.join(dst_path, f"diffusion_{ix}.png"))
        plt.close()

def get_plot_figure(sampled_images):
    sampled_images = torch.cat([i for i in sampled_images.cpu()], dim=-1).permute(1,2,0)
    fig = plt.figure(figsize=(10, 2))
    plt.imshow(sampled_images)
    plt.xticks([]); plt.yticks([]); plt.box(False)
    fig.tight_layout()
    return fig    
