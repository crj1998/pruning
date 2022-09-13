from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MoveArguments():
    mask_learning_rate: float = field(default=0.1, metadata={"help": "Learning rate for regularization."})
    freeze_embeddings: bool = field(default=False, metadata={"help": "Whether we should freeze the embeddings."})

    prepruning_finetune_epochs: int = field(default=1, metadata={"help": "Finetuning epochs before pruning"})

    initial_threshold: float = field(default=1.0, metadata={"help": "Start sparsity (pruned percentage)"})
    final_threshold: float = field(default=0.1, metadata={"help": "Epsilon for sparsity"})
    initial_warmup: int = field(default=1, metadata={"help": "Run `initial_warmup` * `warmup_steps` steps of threshold warmup during which threshold stays at its `initial_threshold` value (sparsity schedule)."})
    final_warmup: int = field(default=2, metadata={"help": "Run `final_warmup` * `warmup_steps` steps of threshold cool-down during which threshold stays at its final_threshold value (sparsity schedule)."})
    final_lambda: float = field(default=0.0, metadata={"help": "Regularization intensity (used in conjunction with `regularization`."})

    regularization: str = field(default='l0', metadata={"help": "use regularization"})
    # distillation setup
    distill_loss_alpha: float = field(default=0.9, metadata={"help": "Distillation loss weight"})
    distill_temp: float = field(default=2./3., metadata={"help": "Distillation temperature"})

    # def __post_init__(self):
    #     if self.pretrained_pruned_model == "None":
    #         self.pretrained_pruned_model = None
    #     if self.pruning_type == "None":
    #         self.pruning_type = None