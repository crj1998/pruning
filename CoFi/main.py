import random
import copy
import numpy as np 

import torch
import torch.nn as nn


from transformers import ViTFeatureExtractor, TrainingArguments, default_data_collator
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Image

from args import CoFiArguments
from trainer import CoFiTrainer


train_ds, test_ds = load_dataset('cifar10', split=['train[:50000]', 'test[:10000]'])
splits = train_ds.train_test_split(test_size=0.1, seed = 42)
train_ds = splits['train']
val_ds = splits['test']

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
data_collator = default_data_collator

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

preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
preprocessed_val_ds = val_ds.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)


from vit import CoFiViTForImageClassification
from l0module import L0Module

# exp_name = "base+distilv12(2-5-8-11,0.5)"
sparsity = 0.70
distill_version = 0
alpha = 0.1
# exp_name = f"base+distilv{distill_version}({alpha})"
exp_name = f"base({sparsity})"
# exp_name = f"base+distil"
# exp_name = f"base(20-40)"


args = TrainingArguments(
    output_dir=f'exp0921/{exp_name}',
    disable_tqdm=False,
    evaluation_strategy="epoch",
    eval_steps=256,
    warmup_steps=512,
    save_strategy="epoch",
    learning_rate=0.00002,
    per_device_train_batch_size=96,
    per_device_eval_batch_size=128,
    dataloader_num_workers=4,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=16,
    logging_first_step=True,
    logging_strategy='steps',
    logging_dir=f'logs/{exp_name}',
    report_to="wandb"
)

cofi_args = CoFiArguments(
    ex_name ="CoFi-ViT-CIFAR10",
    pruning_type="structured_heads+structured_mlp+hidden+layer",
    target_sparsity=sparsity,
    start_sparsity=0.0,
    layer_distill_version=distill_version,
    distill_loss_alpha=1.0-alpha, 
    distill_ce_loss_alpha=alpha,
    distill_temp=2.0,
    # dev
    lagrangian_warmup_epochs=8
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)

model = CoFiViTForImageClassification.from_pretrained('exp/finetuned-cifar10/checkpoint-4690')
teacher_model = copy.deepcopy(model)

config = model.config
l0module = L0Module(
    config, 
    start_sparsity=cofi_args.start_sparsity, 
    target_sparsity=cofi_args.target_sparsity,
)


trainer = CoFiTrainer(
    model, args, cofi_args,
    train_dataset=preprocessed_train_ds,
    eval_dataset=preprocessed_val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    l0_module=l0module, 
    teacher_model=teacher_model
)

trainer.train()

outputs = trainer.predict(preprocessed_test_ds)
print(outputs.metrics)

import os
output_dir = f'exp0921/{exp_name}'

torch.save(l0module.state_dict(), os.path.join(output_dir, "l0_module.pt"))
torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
"""
CUDA_VISIBLE_DEVICES=4 python main.py

"""