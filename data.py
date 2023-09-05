#%%
import pandas as pd
path = "/shared/datasets/TMNIST/TMNIST_Data.csv"
df = pd.read_csv(path)


# %%

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch

N = 8

stacked = []
for i in range(N*N):
    row = df.sample()
    stacked.append(row.iloc[:, 2:].values.reshape(1, 28, 28))

batched_tensor = torch.tensor(np.stack(stacked))


grid_img = make_grid(batched_tensor, nrow=5)

plt.imshow(grid_img.permute(1, 2, 0))


# %%
