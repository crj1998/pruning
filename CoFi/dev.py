import torch

from l0module import L0Module
from vit import CoFiViTForImageClassification

x = torch.rand(1, 3, 224, 224)


# model = CoFiViTForImageClassification.from_pretrained('test-cifar-10/checkpoint-1056')
model = CoFiViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

l0module = L0Module(model.config)
zs = l0module(True)

with torch.no_grad():
    y = model(x, **zs)
    # y = model(x)
    print(y)