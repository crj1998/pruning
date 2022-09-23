
import torch
from transformers.models.vit.modeling_vit import ViTForImageClassification
from l0module import L0Module

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.eval().to(device)
print(model)
# config = model.config
# l0_module = L0Module(config)
# l0_module.load_state_dict(torch.load("/root/rjchen/workspace/CoFi/exp/base/l0_module.pt"))

# zs = l0_module.forward(training=False)
# print(zs)


