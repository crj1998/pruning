from email.policy import strict
import random

import numpy as np 

import torch
import torch.nn as nn


from transformers import ViTFeatureExtractor, TrainingArguments, Trainer, default_data_collator, ViTForImageClassification
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Image




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

    return examples


features = Features({
    'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
    'img': Image(decode=True, id=None),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})

preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
preprocessed_val_ds = val_ds.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)


from vit0 import MoveViTForImageClassification
from transformers import ViTConfig
config = ViTConfig.from_pretrained('google/vit-base-patch16-224')

del config.label2id
del config.id2label
config.pruning_method = "topK"
config.mask_init = "constant"
config.mask_scale = 0.0
config.num_labels = 10

teacher_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True)
model = MoveViTForImageClassification(config)
model.load_state_dict(teacher_model.state_dict(), strict=False)

args = TrainingArguments(
    f"exp/finetuned-cifar10",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=96,
    per_device_eval_batch_size=512,
    dataloader_num_workers=4,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs/finetuned',
    logging_steps=16,
    logging_first_step=True,
    logging_strategy='steps',
    report_to="tensorboard"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)


trainer = Trainer(
    model, args,
    train_dataset=preprocessed_train_ds,
    eval_dataset=preprocessed_val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

outputs = trainer.predict(preprocessed_test_ds)
print(outputs.metrics)
"""
CUDA_VISIBLE_DEVICES=3 python finetune.py
"""