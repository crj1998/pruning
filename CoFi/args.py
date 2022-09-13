from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CoFiArguments():
    test: bool = field(
        default=False,
        metadata={
            "help": "Testing additional arguments."
        },
    )

    ex_name: str = field(default="test", metadata={"help": "Name of experiment. Base directory of output dir."})
    pruning_type: str = field(default=None, metadata={"help": "Type of pruning"})
    reg_learning_rate: float = field(default=0.1, metadata={"help": "Learning rate for regularization."})
    scheduler_type: str = field(default="linear", metadata={"help": "type of scheduler"})
    freeze_embeddings: bool = field(default=False, metadata={"help": "Whether we should freeze the embeddings."})

    pretrained_pruned_model: str = field(default=None, metadata={"help": "Path of pretrained model."})

    droprate_init: float = field(default=0.5, metadata={"help": "Init parameter for loga"})
    temperature: float = field(default=2./3., metadata={"help": "Temperature controlling hard concrete distribution"})
    prepruning_finetune_epochs: int = field(default=1, metadata={"help": "Finetuning epochs before pruning"})
    lagrangian_warmup_epochs: int = field(default=2, metadata={"help": "Number of epochs for lagrangian warmup"})
    target_sparsity: float = field(default=0, metadata={"help": "Target sparsity (pruned percentage)"})
    start_sparsity: float = field(default=0, metadata={"help": "Start sparsity (pruned percentage)"})
    sparsity_epsilon: float = field(default=0, metadata={"help": "Epsilon for sparsity"})

    # distillation setup
    distillation_path: str = field(default=None, metadata={"help": "Path of the teacher model for distillation."})
    do_distill: bool = field(default=False, metadata={"help": "Whether to do distillation or not, prediction layer."})
    do_layer_distill: bool = field(default=False, metadata={"help": "Align layer output through distillation"})
    layer_distill_version: int = field(default=1, metadata={"help": "1: add loss to each layer, 2: add loss to existing layers only"})
    distill_loss_alpha: float = field(default=0.9, metadata={"help": "Distillation loss weight"})
    distill_ce_loss_alpha: float = field(default=0.1, metadata={"help": "Distillation cross entrypy loss weight"})
    distill_temp: float = field(default=2./3., metadata={"help": "Distillation temperature"})

    def __post_init__(self):
        if self.pretrained_pruned_model == "None":
            self.pretrained_pruned_model = None
        if self.pruning_type == "None":
            self.pruning_type = None