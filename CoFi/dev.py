from trainer import CoFiTrainer
from l0module import L0Module
from vit import CoFiViTForImageClassification

import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import ViTFeatureExtractor, default_data_collator, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Image

from transformers.trainer_utils import RemoveColumnsCollator


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def preprocess_images(examples):
    images = examples['img']
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image in images]
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']

    return {'pixel_values': inputs['pixel_values'], 'label': examples['label']}


features = Features({
    'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
    'img': Image(decode=True, id=None),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})

train_ds, test_ds = load_dataset('cifar10', split=['train[:50000]', 'test[:10000]'])
dataset = test_ds.map(preprocess_images, batched=True, features=features)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)

# model = CoFiViTForImageClassification.from_pretrained('exp/finetuned-cifar10/checkpoint-4690')
# model.eval()


exp_name = "base+distilv2"
model = CoFiViTForImageClassification.from_pretrained('exp/finetuned-cifar10/checkpoint-4690')
model.load_state_dict(torch.load(f"exp/{exp_name}/model.pt"))
model.eval()

l0module = L0Module(model.config)
l0module.load_state_dict(torch.load(f"exp/{exp_name}/l0_module.pt"))
zs = l0module.forward(False)
results = l0module.calculate_model_size(zs)
sparsity = results["pruned_model_sparsity"]


args = TrainingArguments(
    f"exp/none",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    dataloader_num_workers=4,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=16,
    logging_first_step=True,
    logging_strategy='steps',
)

trainer = CoFiTrainer(
    model, args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    l0_module=l0module, 
)

# dataloader = trainer.get_eval_dataloader(dataset)
# import inspect
# signature = inspect.signature(model.forward)
# _signature_columns = list(signature.parameters.keys())
# # Labels may be named label or label_ids, the default data collator handles that.
# _signature_columns += list(set(["label", "label_ids"]))

# data_collator = RemoveColumnsCollator(default_data_collator, _signature_columns)
# dataloader = DataLoader(
#     dataset,
#     batch_size=100, 
#     collate_fn=data_collator,
#     drop_last=False,
#     num_workers=4,
#     pin_memory=True,
# )
# print(dataloader)
outputs = trainer.predict(dataset)
# print(outputs.test_accuracy)
print(outputs.metrics)

