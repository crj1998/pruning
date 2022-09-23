import math
import os
import sys
import time
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
from transformers.trainer_utils import (EvalPrediction, PredictionOutput, TrainOutput)
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.training_args import TrainingArguments

from args import CoFiArguments
from l0module import L0Module

class CoFiTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel = None,
        args: TrainingArguments = None,
        cofi_args: CoFiArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        l0_module: Optional[L0Module]=None, 
        teacher_model: Optional[PreTrainedModel]=None, 
        **kwargs
    ):

        Trainer.__init__(self, model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

        self.cofi_args = cofi_args

        self.l0_module = l0_module
        self.prepruning_finetune_steps = 32
        self.start_prune = False
        self.zs = None
        self.l0_optimizer = None
        self.lagrangian_optimizer = None

        self.start_saving_best = True if self.cofi_args and self.cofi_args.pruning_type is None else False

        self.teacher_model = teacher_model
        self.teacher_layer_trans = None
        self.student_layer_trans = None
        if self.teacher_model is not None:
            self.teacher_layer_trans = nn.Linear(model.config.hidden_size, 10).to(self.args.device)
            self.student_layer_trans = nn.Linear(model.config.hidden_size, 10).to(self.args.device)
            self.teacher_model = self.teacher_model.to(self.args.device)
            self.teacher_model.eval()
        
        if self.cofi_args:
            assert self.cofi_args.layer_distill_version == 0 or (self.cofi_args.layer_distill_version > 0 and self.teacher_model is not None)


    def create_optimizer_and_scheduler(self, num_training_steps: int, build_l0_optimizer:bool=True):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight", "CoFiLayerNorm.weight"]
            freeze_keywords = ["embeddings"]
            trans_params = []
            if self.student_layer_trans is not None:
                trans_params.extend(list(self.student_layer_trans.parameters()))
            if self.teacher_layer_trans is not None:
                trans_params.extend(list(self.teacher_layer_trans.parameters()))
            main_model_params = [{
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)] + trans_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                }, {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                }
            ]

            self.optimizer = AdamW(
                main_model_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

            if build_l0_optimizer and self.l0_module is not None:
                l0_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" not in n],
                    "weight_decay": 0.0,
                    "lr": self.cofi_args.reg_learning_rate
                }]

                self.l0_optimizer = AdamW(
                    l0_params,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon
                )

                lagrangian_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                    "weight_decay": 0.0,
                    "lr": - self.cofi_args.reg_learning_rate
                }]

                self.lagrangian_optimizer = AdamW(
                    lagrangian_params,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon
                )

        if self.lr_scheduler is None:
            if self.cofi_args.scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = None

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs):
        if self.cofi_args.do_layer_distill <= 1 or not self.start_prune:
            return None
        mse_loss = nn.MSELoss(reduction="mean")
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        head_layer_z = mlp_z = None
        if "mlp_z" in zs:
            mlp_z = zs["mlp_z"].detach().cpu()
        if "head_layer_z" in zs:
            head_layer_z = zs["head_layer_z"].detach().cpu()
    

        teacher_layer_output = teacher_outputs.hidden_states[1:]
        student_layer_output = student_outputs.hidden_states[1:]
 
        layer_loss = torch.tensor(0.0).to(self.args.device)
        # distilliting existing layers
        if self.cofi_args.layer_distill_version == 2:
            cnt = 0
            for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                if mlp_z[layer_num] > 0:
                    layer_loss += mse_loss(t_layer_o, self.model.layer_transformation(s_layer_o))
                    cnt += 1
            layer_loss = layer_loss / max(cnt, 1)

        # distilling layers with a minimal distance
        elif self.cofi_args.layer_distill_version > 2:
            l = []
            # specified_teacher_layers = [2, 5, 8, 11]
            specified_teacher_layers = [2, 5, 8, 11]
            transformed_s_layer_o = [self.model.layer_transformation(s_layer_o) for s_layer_o in student_layer_output]
            specified_teacher_layer_reps = [teacher_layer_output[i] for i in specified_teacher_layers] #! teacher: 4x[32,113,768]

            for t_layer_o in specified_teacher_layer_reps:
                for i, s_layer_o in enumerate(transformed_s_layer_o): #! student: 12x[32,113,768]
                    l.append(mse_loss(t_layer_o, s_layer_o))
            layerwiseloss = torch.stack(l).reshape(len(specified_teacher_layer_reps), len(student_layer_output)) #! [4,12]

            existing_layers = None
            if head_layer_z is not None:
                existing_layers = (head_layer_z.to(self.args.device) != 0)

            #! no ordering restriction specified
            if self.cofi_args.layer_distill_version == 3:
                alignment = torch.argmin(layerwiseloss, dim=1)
            #! added the ordering restriction -> to choose the min loss in 4 student layers
            elif self.cofi_args.layer_distill_version == 4:
                last_aligned_layer = 12
                alignment = []
                for search_index in range(3, -1, -1):
                    indexes = layerwiseloss[search_index].sort().indices
                    if existing_layers is not None:
                        align = indexes[(indexes < last_aligned_layer) & existing_layers]
                    else:
                        align = indexes[indexes < last_aligned_layer]
                    if len(align) > 0:
                        align = align[0]
                    else:
                        align = last_aligned_layer
                    alignment.append(align)
                    last_aligned_layer = align
                alignment.reverse()
                alignment = torch.tensor(alignment, device=self.args.device)
            else:
                raise ValueError(f"{self.cofi_args.layer_distill_version} version is not specified.")

            layerwise = torch.arange(len(specified_teacher_layers), device=self.args.device)
            #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
            # layer_loss += layerwiseloss[layerwise, alignment].sum()
            layer_loss += layerwiseloss[layerwise, alignment].mean()

            # if self.global_step % 100 == 0:
            #     print(f"v{self.cofi_args.layer_distill_version} Global step: {self.global_step}, Alignment: " + str(alignment))
        elif self.cofi_args.layer_distill_version > 10:
            layerwiseloss = []
            specified_layers = [2, 5, 8, 11]
            if self.cofi_args.layer_distill_version == 11:
                specified_student_out = [student_layer_output[i] for i in specified_layers]
                specified_teacher_out = [self.model.layer_transformation(teacher_layer_output[i]) for i in specified_layers] #! teacher: 4x[32,113,768]
                for s_out, t_out in zip(specified_student_out, specified_teacher_out):
                    layerwiseloss.append(mse_loss(s_out, t_out))
                layerwiseloss = torch.stack(layerwiseloss)
                layer_loss += layerwiseloss.mean()

            if self.cofi_args.layer_distill_version == 12:
                specified_student_out = [self.student_layer_trans(student_layer_output[i]) for i in specified_layers]
                specified_teacher_out = [self.teacher_layer_trans(teacher_layer_output[i]) for i in specified_layers]
                for s_out, t_out in zip(specified_student_out, specified_teacher_out):
                    layerwiseloss.append(kl_loss(F.log_softmax(s_out, dim=-1), F.softmax(t_out, dim=-1)))
                layerwiseloss = torch.stack(layerwiseloss)
                layer_loss += layerwiseloss.mean()

            if self.cofi_args.layer_distill_version == 13:
                encoder_layer_idxs = []
                layer_loss = 0.0
                for i in range(12):
                    if mlp_z[i] > 0.0  and head_layer_z > 0.0:
                        encoder_layer_idxs.append(i)
                for i, idx in enumerate(encoder_layer_idxs):
                    layer_disitil_loss = []
                    for j in range(idx, 12):
                        layer_disitil_loss.append(F.mse_loss(student_layer_output[i], teacher_layer_output[j]))
                    layer_loss += min(layer_disitil_loss)

        return layer_loss

    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs):
        distill_layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs)

        distill_logit_loss = F.kl_div(
            input = F.log_softmax(student_outputs.logits / self.cofi_args.distill_temp, dim=-1), #! logits: [32,3]
            target = F.softmax(teacher_outputs.logits / self.cofi_args.distill_temp, dim=-1), #! distill_temp: 2.0
            reduction="batchmean"
        ) * (self.cofi_args.distill_temp ** 2)

        distill_loss = self.cofi_args.distill_ce_loss_alpha * distill_logit_loss
        if distill_layer_loss is not None:
            distill_loss += self.cofi_args.distill_loss_alpha * distill_layer_loss

        return distill_loss, distill_layer_loss, distill_logit_loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()
        if self.l0_module is not None:
            self.l0_module.train()

        inputs = self._prepare_inputs(inputs)

        distil_loss = distil_layer_loss = distil_logit_loss = lagrange = None
        loss = 0.0
        
        if self.cofi_args.layer_distill_version == 0:
            distil_loss = self.compute_loss(model, inputs)
        else:
            with torch.no_grad():
                # only retain inputs of certain keys
                teacher_inputs_keys = ["labels", "pixel_values"]
                teacher_inputs = {key: inputs[key] for key in teacher_inputs_keys if key in inputs}
                teacher_outputs = self.teacher_model(**teacher_inputs, output_attentions=False, output_hidden_states=True)
            student_outputs = model(**inputs, output_attentions=False, output_hidden_states=True)

            zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs
            distil_loss, distil_layer_loss, distil_logit_loss = self.calculate_distillation_loss(teacher_outputs, student_outputs, zs)
        
        loss += distil_loss
        if self.start_prune and self.l0_module is not None:
            lagrange, *_ = self.l0_module.lagrangian_regularization(self.global_step - self.prepruning_finetune_steps)
            loss += lagrange

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return {
            "loss": loss.detach(),
            "lagrange_loss": lagrange.detach() if lagrange is not None else None,
            "distil_loss": distil_loss.detach() if distil_loss is not None else None,
            "distil_layer_loss": distil_layer_loss.detach() if distil_layer_loss is not None else None,
            "distil_logit_loss": distil_logit_loss.detach() if distil_logit_loss is not None else None
        }

    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.l0_module is not None:
            lagrangian_warmup_steps = self.cofi_args.lagrangian_warmup_epochs * num_update_steps_per_epoch
            self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)

        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(self.args.max_steps % num_update_steps_per_epoch > 0)
        else:
            self.t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = self.t_total

        self.create_optimizer_and_scheduler(num_training_steps=self.t_total, build_l0_optimizer = self.start_prune)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0

        epochs_trained = 0

        model = self.model
        model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()

        self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        metrics = self.evaluate()

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=disable_tqdm, ncols=120)

        # training
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

            total_loss = torch.tensor(0.0).to(self.args.device)
            lagrange_loss = torch.tensor(0.0).to(self.args.device)
            distil_loss = torch.tensor(0.0).to(self.args.device)

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
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)

                if self.start_prune and self.l0_module is not None:
                    zs = self.l0_module.forward(training=True)
                    inputs.update(zs)

                loss_dict = self.training_step(model, inputs)

                total_loss += loss_dict["loss"]
                lagrange_loss += (loss_dict["lagrange_loss"] or 0.0)
                distil_loss += (loss_dict["distil_loss"] or 0.0)

                # self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps and (step + 1) == len(epoch_iterator)
                ):
                    nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()

                    if self.l0_module is not None and self.l0_optimizer is not None:
                        self.l0_optimizer.step()
                        self.lagrangian_optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self.l0_module is not None:
                        self.l0_module.constrain_parameters()

                    model.zero_grad()
                    if self.l0_module is not None:
                        self.l0_module.zero_grad()

                    self.optimizer.zero_grad()
                    if self.l0_optimizer is not None:
                        self.l0_optimizer.zero_grad()
                    if self.lagrangian_optimizer is not None:
                        self.lagrangian_optimizer.zero_grad()

                    self.global_step += 1
                    self.state.global_step = self.global_step
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else self.args.learning_rate
                    logs: Dict[str, float]= {
                        'lr': lr,
                        'loss': loss_dict["loss"].item(),
                        'distil': loss_dict["distil_loss"].item() if loss_dict["distil_loss"] is not None else 0.0,
                        'lagrange': loss_dict["lagrange_loss"].item() if loss_dict["lagrange_loss"] is not None else 0.0,
                    }
                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (self.global_step == 1 and self.args.logging_first_step):
                        pruned_sparsity = self.l0_module.calculate_model_size(zs)['pruned_model_sparsity'] if self.start_prune else 0.0
                        logs.update({
                            'lam1': self.l0_module.lambda_1.item(),
                            'lam2': self.l0_module.lambda_2.item(),
                            'sparsity': round(pruned_sparsity, 4)
                        })
                        self.log(logs)
                    # if self.global_step % self.args.eval_steps == 0:
                    #     self.evaluate()

                    epoch_pbar.set_description_str(f"Train({self.epoch:.2f})")
                    # epoch_pbar.set_postfix(logs)
                    epoch_pbar.set_postfix_str(f"lr={logs['lr']:.6f}, loss={logs['loss']:.3f}, lagrange={logs['lagrange']:.3f}, distil={logs['distil']:.3f}")
                    epoch_pbar.update(self.args.gradient_accumulation_steps)

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

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        pruned_sparsity = 1.0
        if self.start_prune:
            self.l0_module.eval()
            self.zs = self.l0_module.forward(training=False)
            pruned_sparsity = self.l0_module.calculate_model_size(self.zs)['pruned_model_sparsity']
        metrics = super().evaluate(*args, **kwargs)

        return {
            'Accuracy': metrics['eval_accuracy'], 
            'Throughput': metrics['eval_samples_per_second'], 
            'Sparsity': round(pruned_sparsity, 4)
        }

    def predict(self,  *args, **kwargs):
        pruned_sparsity = 1.0
        if self.start_prune:
            self.l0_module.eval()
            self.zs = self.l0_module.forward(training=False)
            pruned_sparsity = self.l0_module.calculate_model_size(self.zs)['pruned_model_sparsity']
        metrics = super().predict(*args, **kwargs)
        metrics.metrics['sparsity'] = round(pruned_sparsity, 4)
        return metrics

    def prediction_step(self, model, inputs, *args, **kwargs):
        if self.zs is not None:
            inputs.update(self.zs)
        return super().prediction_step(model, inputs, *args, **kwargs)
    