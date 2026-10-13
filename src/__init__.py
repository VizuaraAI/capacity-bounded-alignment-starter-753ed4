"""Capacity bounded alignment: is the Superficial Alignment Hypothesis scale bounded?

Module map
----------
model_zoo    the capacity ladder, defined once
data         dataset loaders for training mixtures and both evaluation axes
prompts      base few shot, in context alignment, and chat template prompting
train_sft    supervised fine tuning entry point (optionally LoRA)
evaluate     scores one model and intervention on the format and capability axes
token_shift  token distribution shift between a base model and its aligned twin
metrics      the Superficiality Index and its bootstrap confidence intervals
analyze      turns a directory of result files into the paper's tables
"""

__version__ = "0.1.0"
