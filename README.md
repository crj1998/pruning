# CoFi for ViT
This repository contains the modified CoFi code for Image Classification task with ViT. 
Original paper:  [Structured Pruning Learns Compact and Accurate Models](https://arxiv.org/abs/2204.00408) @ ACL'22

Offical code: [CoFi](https://github.com/princeton-nlp/CoFiPruning)


## Finetuning before pruning
`google/vit-base-patch16-224`
Performance on ImageNet-1k Validation: Top1@Acc:81.66% Top5@Acc:96.09%

Finetuned on cifar10:
    optimizer: Adam, lr = 0.00002 weight decay: 0.01
    batch_size: 96 epochs: 10
    performance: Acc on cifar-10 test: **98.37**%


## CoFi Pruning

