---
license: apache-2.0
base_model: EleutherAI/polyglot-ko-5.8b
tags:
- generated_from_trainer
model-index:
- name: polyglot-5.8b-koalpaca-v1.1a-rtx3090
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# polyglot-5.8b-koalpaca-v1.1a-rtx3090

This model is a fine-tuned version of [EleutherAI/polyglot-ko-5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 128
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.40.0
- Pytorch 2.2.1+cu118
- Datasets 2.19.0
- Tokenizers 0.19.1
