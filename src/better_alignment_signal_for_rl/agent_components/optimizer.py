# Copyright 2022 by Imperium Global Ventures Ltd.
# All rights reserved.
# This file is part of the AVM Toy Model Project,
# and is released under the "Apache License 2.0". Please see the LICENSE
# file that should have been included as part of this package.
import json
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        a function that produce learning rate given step，
        noted that before the training, pytorch will call lr_scheduler.step() once
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # during warmup lr goes from warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # after warmup, lr goes from 1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # record optimize training params
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # record params names
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def get_optims_and_scheduler(model, optimizer_cls, wd, lr, n_epochs, steps_per_epoch):
    pg = get_params_groups(model, weight_decay=wd)
    
    if optimizer_cls is not None:
        optimizer = getattr(optim, optimizer_cls)(pg, lr=lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(pg, lr=lr, weight_decay=wd)

    lr_scheduler = create_lr_scheduler(optimizer, steps_per_epoch, n_epochs,
                                       warmup=True, warmup_epochs=1)
    return optimizer, lr_scheduler

