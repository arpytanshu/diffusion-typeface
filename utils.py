import os
import sys
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32,32))
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
