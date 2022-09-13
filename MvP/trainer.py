import os, sys, time, math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_utils import (EvalPrediction, TrainOutput)
from transformers.training_args import TrainingArguments

from args import MoveArguments



def schedule_threshold(
    step: int, total_step: int,
    warmup_steps: int,
    initial_threshold: float,
    final_threshold: float,
    initial_warmup: int,
    final_warmup: int,
    final_lambda: float,
):
    if step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif step > (total_step - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff**3)
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda


def regularization(model: nn.Module, mode: str):
    reg, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                reg += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                reg += torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum() / param.numel()
            else:
                raise ValueError("Don't know this mode.")
            counter += 1
    return reg / counter

class MoveTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel = None,
        args: TrainingArguments = None,
        move_args: MoveArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        teacher_model: Optional[PreTrainedModel]=None, 
        **kwargs
    ):

        Trainer.__init__(self, model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

        self.move_args = move_args

        self.prepruning_finetune_steps = 16
        self.start_prune = False

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.args.device)
            self.teacher_model.eval()

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            freeze_keywords = ["embeddings"]

            grouped_parameters = [{
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if ("mask_scores" in n) and not any(fk in n for fk in freeze_keywords)
                ],
                "lr": self.move_args.mask_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if ("mask_scores" not in n) and not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)
                ],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if ("mask_scores" not in n) and any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)
                ],
                "lr": self.args.learning_rate,
                "weight_decay": 0.0,
            }]

            self.optimizer = AdamW(
                grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()

        inputs = self._prepare_inputs(inputs)
        # only retain inputs of certain keys
        inputs_keys = ["labels", "pixel_values"]
        inputs = {key: inputs[key] for key in inputs_keys if key in inputs}

        loss = reg_loss = distil_loss = None

        threshold, regu_lambda = schedule_threshold(
            step=self.global_step,
            total_step=self.t_total,
            warmup_steps=self.args.warmup_steps,
            final_threshold=self.move_args.final_threshold,
            initial_threshold=self.move_args.initial_threshold,
            final_warmup=self.move_args.final_warmup,
            initial_warmup=self.move_args.initial_warmup,
            final_lambda=self.move_args.final_lambda,
        )

        if not self.start_prune:
            threshold = 1.0
            regu_lambda = 0.0

        if self.teacher_model is not None :
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs, output_attentions=True, output_hidden_states=True)

            inputs.update({'threshold': threshold})
            student_outputs = model(**inputs, output_attentions=True, output_hidden_states=True, threshold=threshold)
            distill_loss = F.kl_div(
                input = F.log_softmax(student_outputs.logits / self.move_args.distill_temp, dim=-1),
                target = F.softmax(teacher_outputs.logits / self.move_args.distill_temp, dim=-1),
                reduction="batchmean"
            ) * (self.move_args.distill_temp ** 2)
            loss = self.move_args.distill_loss_alpha * distill_loss + (1.0 - self.move_args.distill_loss_alpha) * student_outputs.loss
        else:
            inputs.update({'threshold': threshold})
            loss = self.compute_loss(model, inputs)

        # Regularization
        if self.start_prune and self.move_args.regularization is not None:
            reg_loss = regularization(model=model, mode=self.move_args.regularization)
            loss += regu_lambda * reg_loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return {
            "loss": loss.detach(),
            "distil_loss": distil_loss.detach() if distil_loss is not None else None,
            "regular_loss": reg_loss.detach() if reg_loss is not None else None,
        }

    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(self.args.max_steps % num_update_steps_per_epoch > 0)
        else:
            self.t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = self.t_total

        self.create_optimizer_and_scheduler(num_training_steps=self.t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0

        epochs_trained = 0

        model = self.model
        model.zero_grad()
        self.optimizer.zero_grad()

        # metrics = self.evaluate()
        # print(metrics)

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=disable_tqdm, ncols=120)

        # training
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

            total_loss = torch.tensor(0.0).to(self.args.device)
            distil_loss = torch.tensor(0.0).to(self.args.device)
            regular_loss = torch.tensor(0.0).to(self.args.device)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = train_dataloader
            epoch_pbar = tqdm(epoch_iterator, desc="Train", disable=disable_tqdm, ncols=120)

            for step, inputs in enumerate(epoch_iterator):
                inputs: dict[str, Any] = inputs
                if self.prepruning_finetune_steps > 0 and self.global_step == self.prepruning_finetune_steps: #! before pruning
                    self.start_prune = True

                    self.optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.t_total - self.global_step

                    # reset the optimizer
                    self.create_optimizer_and_scheduler(lr_steps)

                loss_dict = self.training_step(model, inputs)

                total_loss += loss_dict["loss"]
                distil_loss += (loss_dict["distil_loss"] or 0.0)
                regular_loss += (loss_dict["regular_loss"] or 0.0)

                # self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps and (step + 1) == len(epoch_iterator)
                ):
                    nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.optimizer.zero_grad()
                
                    lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else self.args.learning_rate
                    logs: Dict[str, float]= {
                        'lr': lr,
                        'loss': loss_dict["loss"].item(),
                        'distil': loss_dict["distil_loss"].item() if loss_dict["distil_loss"] is not None else 0.0,
                        'regular': loss_dict["regular_loss"].item() if loss_dict["regular_loss"] is not None else 0.0,
                    }
                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (self.global_step == 1 and self.args.logging_first_step):
                        logs.update({
                            # 'sparsity': round(pruned_sparsity, 4)
                        })
                        self.log(logs)
                    # if self.global_step % self.args.eval_steps == 0:
                    #     self.evaluate()

                    epoch_pbar.set_description_str(f"Train({self.epoch:.2f})")
                    epoch_pbar.set_postfix_str(f"lr={logs['lr']:.6f}, loss={logs['loss']:.3f}, regular={logs['regular']:.3f}, distil={logs['distil']:.3f}")
                    epoch_pbar.update(self.args.gradient_accumulation_steps)

                self.global_step += 1
                self.state.global_step = self.global_step
                self.epoch = epoch + (step + 1) / len(epoch_iterator)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

            epoch_pbar.close()

            final_loss = total_loss.item()/(step+1)
            metrics = self.evaluate()
            train_pbar.set_postfix(metrics)
            train_pbar.update(1)

        train_pbar.close()

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        return TrainOutput(self.global_step, final_loss, None)

