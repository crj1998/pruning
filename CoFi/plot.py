import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import torch

from l0module import L0Module
from vit0 import CoFiViTForImageClassification


exp_name = "base(0.1)"
model = CoFiViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
l0module = L0Module(model.config)
l0module.load_state_dict(torch.load(f"exp/{exp_name}/l0_module.pt"))
data = l0module.forward(False)
results = l0module.calculate_model_size(data)
sparsity = results["pruned_model_sparsity"]


# 定义colorbar的颜色帧
color_list = ['pink', 'blue']

# 线性补帧，并定义自定义colormap的名字，此处为rain
my_cmap = LinearSegmentedColormap.from_list('custom', color_list)

# 注册自定义的cmap，此后可以像使用内置的colormap一样使用自定义的rain
cm.register_cmap(cmap=my_cmap)

fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(10, 4), dpi=120, gridspec_kw={'width_ratios': [1.15, 3]})

# data['head_layer_z'].detach().squeeze().numpy().reshape(-1, 1)
x = data['head_z'].detach().squeeze().numpy()
ax[0].matshow(x, cmap="custom", vmin=0.0, vmax=1.0)
ax[0].set_xlabel("Heads")
ax[0].set_ylabel("Layer")
ax[0].set_xticks([i for i in range(12)], [str(i+1) for i in range(12)])
ax[0].set_yticks([i for i in range(12)], [str(i+1) for i in range(12)])
# Minor ticks
ax[0].set_xticks([i-0.5 for i in range(12)], minor=True)
ax[0].set_yticks([i-0.5 for i in range(12)], minor=True)
ax[0].xaxis.tick_bottom()
ax[0].tick_params('both', length=0, width=0, which='both')

# Gridlines based on minor ticks
ax[0].grid(which='minor', color='w', linestyle='-', linewidth=1)
ax[0].set_title('MHAs')

x = data['mlp_z'].detach().squeeze().numpy().reshape(-1, 1)*data['intermediate_z'].detach().squeeze().numpy()
x = x.repeat(100, axis=0)
ax[1].matshow(x, cmap="custom", vmin=0.0, vmax=1.0)
ax[1].set_xlabel("Intermediate")

ax[1].set_xticks([i*768 for i in range(1, 5)], [f'{i}.0x' for i in range(1, 5)])
ax[1].set_yticks([i*100+50 for i in range(12)], [str(i+1) for i in range(12)])
ax[1].set_yticks([i*100 for i in range(12)], minor=True)

# Minor ticks

ax[1].xaxis.tick_bottom()
ax[1].yaxis.tick_right()

ax[1].tick_params('both', length=0, width=0, which='both')

# Gridlines based on minor ticks
ax[1].grid(which='minor', axis='y', color='w', linestyle='-', linewidth=1)
ax[1].set_title('FFNs')

fig.tight_layout()

pink_patch = mpatches.Patch(color='pink', label='pruned')
blue_patch = mpatches.Patch(color='blue', label='remain')

fig.legend(handles=[pink_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.4, 0.15), ncol=2)
fig.suptitle(exp_name)

plt.savefig(exp_name.split("/")[-1].replace(".", "") + ".png")
plt.close()