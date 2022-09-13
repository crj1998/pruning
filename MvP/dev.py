
import random
import copy
import numpy as np 

import torch
import torch.nn as nn


from transformers import ViTFeatureExtractor, TrainingArguments, default_data_collator, ViTForImageClassification
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Image

from args import MoveArguments
from trainer import MoveTrainer
from vit import MoveViTForImageClassification


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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



exp_name = "base"
args = TrainingArguments(
    output_dir=f'exp/{exp_name}',
    disable_tqdm=False,
    evaluation_strategy="epoch",
    eval_steps=256,
    warmup_steps=128,
    save_strategy="epoch",
    learning_rate=0.00002,
    per_device_train_batch_size=96,
    per_device_eval_batch_size=128,
    dataloader_num_workers=4,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=16,
    logging_first_step=True,
    # logging_strategy='steps',
    # logging_dir=f'logs/{exp_name}',
    # report_to="wandb"
    report_to='tensorboard'
)

cofi_args = MoveArguments(
    final_threshold=0.1,
    final_lambda=1.0,
    initial_warmup=2,
    final_warmup=5,
    distill_loss_alpha=0.5,
    distill_temp=1.0,
    regularization='l0'
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)

# teacher_model = ViTForImageClassification.from_pretrained('../CoFi/exp/finetuned-cifar10/checkpoint-4690')
# config = teacher_model.config
# config.pruning_method = "topK"
# config.mask_init = "constant"
# config.mask_scale = 0.0

teacher_model = None



model = MoveViTForImageClassification.from_pretrained("exp/finetuned-cifar10/checkpoint-938")
# model.load_state_dict(teacher_model.state_dict())
trainer = MoveTrainer(
    model, args, cofi_args,
    train_dataset=preprocessed_train_ds,
    eval_dataset=preprocessed_val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    teacher_model=teacher_model
)

trainer.train()

outputs = trainer.predict(preprocessed_test_ds)
print(outputs.metrics)

"""
CUDA_VISIBLE_DEVICES=7 python dev.py
"""