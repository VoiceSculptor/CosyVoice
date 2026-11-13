# -*- coding: utf-8 -*-
"""Lightweight LoRA injection for Qwen2 Linear layers (no peft wrapper)."""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale

    def merge(self):
        with torch.no_grad():
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scale
            self.base.weight.data += delta.to(self.base.weight.dtype)
        return self.base


TARGET_SUFFIX = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                 'gate_proj', 'up_proj', 'down_proj')


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32):
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(TARGET_SUFFIX):
            continue
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        for p in module.parameters():
            p.requires_grad_(False)
        setattr(parent, child_name, LoRALinear(module, r, alpha))
        replaced += 1
    for n, p in model.named_parameters():
        p.requires_grad_('lora_' in n)
    return replaced


def merge_lora(model: nn.Module):
    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.merge())
            merged += 1
    return merged


def count_trainable(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total