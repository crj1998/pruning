
import time

from tqdm import tqdm
from copy import deepcopy
# import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from fvcore import nn as fvnn


from l0module import L0Module
from vit import CoFiViTForImageClassification

# device = torch.device("cuda")
device = torch.device("cpu")


class Zeros(nn.Identity):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return (torch.zeros_like(x), )

class Residual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_tensor):
        return input_tensor


@torch.no_grad()
def prune_vit(model, zs):
    model = deepcopy(model)
    state_dict = model.state_dict()
    config = model.config

    hidden_z = zs['hidden_z'].detach()
    head_z = zs['head_z'].detach().squeeze()
    intermediate_z = zs['intermediate_z'].detach().squeeze()
    head_layer_z = zs['head_layer_z'].detach().squeeze()
    mlp_z = zs['mlp_z'].detach().squeeze() 

    hidden_size = (hidden_z>0.0).sum().item()
    num_heads = [(i>0.0).sum().item() for i in head_z]
    intermediates = [(i>0.0).sum().item() for i in intermediate_z]
    mha_layer = [i.item()>0.0 for i in head_layer_z]
    mlp_layer = [i.item()>0.0 for i in mlp_z]

    # prepare model and init
    
    model.vit.embeddings.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
    key = 'vit.embeddings.cls_token'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0] * hidden_z[hidden_z>0.0]
    
    num_patches = model.vit.embeddings.patch_embeddings.num_patches
    model.vit.embeddings.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
    key = 'vit.embeddings.position_embeddings'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0] * hidden_z[hidden_z>0.0]

    model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(config.num_channels, hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
    key = 'vit.embeddings.patch_embeddings.projection.weight'
    value = state_dict[key]
    state_dict[key] = value[hidden_z > 0.0]*hidden_z[hidden_z > 0.0].reshape(-1, 1, 1, 1)
    key = 'vit.embeddings.patch_embeddings.projection.bias'
    value = state_dict[key]
    state_dict[key] = value[hidden_z > 0.0]*hidden_z[hidden_z > 0.0]

    for layer, module in enumerate(model.vit.encoder.layer):
        num_attention_heads = num_heads[layer]
        intermediate_size = intermediates[layer]
        mha = mha_layer[layer]
        mlp = mlp_layer[layer]

        if mha:
            module.layernorm_before = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
            key = f'vit.encoder.layer.{layer}.layernorm_before.weight'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]
            key = f'vit.encoder.layer.{layer}.layernorm_before.bias'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]

            module.attention.attention.num_attention_heads = num_attention_heads
            attention_head_size = module.attention.attention.attention_head_size

            head_dim = head_z[layer].repeat_interleave(attention_head_size)
            

            # module.attention.attention.attention_head_size = config.hidden_size // config.num_attention_heads
            all_head_size = num_attention_heads * module.attention.attention.attention_head_size
            module.attention.attention.all_head_size = all_head_size
            module.attention.attention.query = nn.Linear(hidden_size, all_head_size, bias=config.qkv_bias)
            module.attention.attention.key = nn.Linear(hidden_size, all_head_size, bias=config.qkv_bias)
            module.attention.attention.value = nn.Linear(hidden_size, all_head_size, bias=config.qkv_bias)

            for k1 in ['query', 'key', 'value']:
                key = f'vit.encoder.layer.{layer}.attention.attention.{k1}.weight'
                value = state_dict[key]
                state_dict[key] = value[head_dim > 0.0][:, hidden_z > 0.0]
                
                key = f'vit.encoder.layer.{layer}.attention.attention.{k1}.bias'
                value = state_dict[key]
                state_dict[key] = value[head_dim > 0.0]

            module.attention.output.dense = nn.Linear(all_head_size, hidden_size)
            key = f'vit.encoder.layer.{layer}.attention.output.dense.weight'
            value = state_dict[key]
            state_dict[key] = head_layer_z[layer] * value[hidden_z > 0.0][: , head_dim > 0.0] * head_dim[head_dim > 0.0] * hidden_z[hidden_z > 0.0].reshape(-1, 1)
            key = f'vit.encoder.layer.{layer}.attention.output.dense.bias'
            value = state_dict[key]
            state_dict[key] = head_layer_z[layer] * value[hidden_z > 0.0] * hidden_z[hidden_z > 0.0]

        else:
            module.layernorm_before = nn.Identity()
            module.attention = Zeros()
            del state_dict[f'vit.encoder.layer.{layer}.layernorm_before.weight']
            del state_dict[f'vit.encoder.layer.{layer}.layernorm_before.bias']
            for k1 in ['attention.query', 'attention.key', 'attention.value', 'output.dense']:
                for k2 in ['weight', 'bias']:
                    del state_dict[f'vit.encoder.layer.{layer}.attention.{k1}.{k2}']
            
        if mlp:
            inter_dim = intermediate_z[layer]

            module.layernorm_after = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
            key = f'vit.encoder.layer.{layer}.layernorm_after.weight'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]
            key = f'vit.encoder.layer.{layer}.layernorm_after.bias'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]

            module.intermediate.dense = nn.Linear(hidden_size, intermediate_size)
            key = f'vit.encoder.layer.{layer}.intermediate.dense.weight'
            value = state_dict[key]
            state_dict[key] = value[inter_dim>0.0][: , hidden_z > 0.0]
            key = f'vit.encoder.layer.{layer}.intermediate.dense.bias'
            value = state_dict[key]
            state_dict[key] = value[inter_dim>0.0]
    
            module.output.dense = nn.Linear(intermediate_size, hidden_size)
            key = f'vit.encoder.layer.{layer}.output.dense.weight'
            value = state_dict[key]
            state_dict[key] = mlp_z[layer] * inter_dim[inter_dim>0.0] * value[hidden_z > 0.0][:, inter_dim > 0.0] * hidden_z[hidden_z > 0.0].reshape(-1, 1)
            key = f'vit.encoder.layer.{layer}.output.dense.bias'
            value = state_dict[key]
            state_dict[key] = mlp_z[layer] * value[hidden_z > 0.0] * hidden_z[hidden_z > 0.0]


        else:
            module.layernorm_after = nn.Identity()
            module.intermediate = nn.Identity()
            module.output = Residual()

            for k1 in ['layernorm_after', 'intermediate.dense', 'output.dense']:
                for k2 in ['weight', 'bias']:
                    del state_dict[f'vit.encoder.layer.{layer}.{k1}.{k2}']

    model.vit.layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
    key = 'vit.layernorm.weight'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0]
    key = 'vit.layernorm.bias'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0]

    model.classifier = nn.Linear(hidden_size, config.num_labels)
    key = "classifier.weight"
    value = state_dict[key]
    state_dict[key] = value[:, hidden_z > 0.0] * hidden_z[hidden_z > 0.0]
    return model, state_dict


IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

test_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

dataset = CIFAR10("../../data", download=False, train=False, transform=test_transform)

dataloader = DataLoader(
    dataset,
    batch_size=100,
    drop_last=False,
    num_workers=4,
    pin_memory=False,
)

exp_name = "exp0921/base(8-20)"
model = CoFiViTForImageClassification.from_pretrained('exp/finetuned-cifar10/checkpoint-4690')
model.load_state_dict(torch.load(f"{exp_name}/model.pt"))
model.eval()

l0module = L0Module(model.config)
l0module.load_state_dict(torch.load(f"{exp_name}/l0_module.pt"))
zs = l0module.forward(False)
results = l0module.calculate_model_size(zs)
sparsity = results["pruned_model_sparsity"]


from transformers.models.vit.modeling_vit import ViTForImageClassification
pruned_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True)

pruned_model.load_state_dict(model.state_dict(), strict=False)
pruned_model, state_dict = prune_vit(pruned_model, zs)
pruned_model.load_state_dict(state_dict)
pruned_model.eval()

model = model.to(device)
pruned_model = pruned_model.to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
del model.layer_transformation
params_base = fvnn.parameter_count(model)['']
params_pruned = fvnn.parameter_count(pruned_model)['']
sparsity = 1 - (params_pruned / params_base)

flops_dict, _ = fvnn.flop_count(model, (dummy_input, ))
flops_base = 0
for name, flops in flops_dict.items():
    flops_base += flops

flops_dict, _ = fvnn.flop_count(pruned_model, (dummy_input, ))
flops_pruned = 0
for name, flops in flops_dict.items():
    flops_pruned += flops
print(flops_pruned, flops_base)
print(sparsity)
print(params_pruned, params_base)

# latency = total = correct = 0
# with torch.no_grad():
#     zs = {k: v.to(device) for k, v in zs.items()}
#     for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Eval", ncols=80):
#         t = time.time()
#         y = model(inputs.to(device), **zs)
#         correct += (y.logits.argmax(dim=-1).cpu() == targets).sum().item()
#         latency += time.time() - t
#         total += targets.size(0)

# print(f"Sparsity: {sparsity:.2%}")
# print(f"Latency: {latency/len(dataset) * 1000:.1f} ms/img")
# print(f"Accuracy: {correct/total:.2%}")


latency = total = correct = 0
with torch.no_grad():
    zs = {k: v.to(device) for k, v in zs.items()}
    for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Eval", ncols=80):
        t = time.time()
        # y = model(inputs.to(device), **zs)
        y = pruned_model(inputs.to(device))
        correct += (y.logits.argmax(dim=-1).cpu() == targets).sum().item()
        latency += time.time() - t
        total += targets.size(0)

print(f"Sparsity: {sparsity:.2%}")
print(f"Latency: {latency/len(dataset) * 1000:.2f} ms/img")
print(f"Accuracy: {correct/total:.2%}")



"""
CUDA_VISIBLE_DEVICES=0 python pruned_eval.py
"""