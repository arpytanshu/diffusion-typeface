
import os
import torch
import fire

from utils import Config, get_labels, get_plot_figure
from modules import Unet
from diffusion import Diffusion
from default_config import config


def sample_diffusion(run_name,  plot_save_path, string='demo', cfg_scale=3):

    args = Config(config=config)

    chkpt = torch.load(os.path.join(run_name, 'models', 'ckpt.pt'))

    model = Unet(device=args.device, num_classes=args.num_classes)
    model.load_state_dict(chkpt['model'])
    model.eval()

    diffusion = Diffusion(img_size=args.img_size, device=args.device)
    labels = torch.tensor(get_labels(string)).long().to(args.device)

    if cfg_scale is None:
        cfg_scale = args.cfg_scale
    
    sampled_images = diffusion.sample(model, len(labels), labels=labels, cfg_scale=cfg_scale)

    fig = get_plot_figure(sampled_images)
    if os.path.isdir(plot_save_path):
       plot_save_path = os.path.join(plot_save_path, 'image.png')
    fig.savefig(plot_save_path)



if __name__ == '__main__':
  fire.Fire(sample_diffusion)
