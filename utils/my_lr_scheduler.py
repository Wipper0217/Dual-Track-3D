import torch
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR)


class StepLRScheduler():
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.scheduler = StepLR(optimizer, step_size, gamma)

    def step(self):
        self.scheduler.step()

    def get_current_lr(self):
        return self.scheduler.get_lr()

class MultiStepLRScheduler():
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.scheduler = MultiStepLR(optimizer, milestones, gamma)

    def step(self):
        self.scheduler.step()

    def get_current_lr(self):
        return self.scheduler.get_lr()

class ExponentialLRScheduler():
    def __init__(self, optimizer, gamma):
        self.scheduler = ExponentialLR(optimizer, gamma)

    def step(self):
        self.scheduler.step()

    def get_current_lr(self):
        return self.scheduler.get_lr()


class CosineAnnealingLRScheduler():
    def __init__(self, optimizer, T_max, eta_min=0):
        self.scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)

    def step(self):
        self.scheduler.step()

    def get_current_lr(self):
        return self.scheduler.get_lr()

LRs = {
    'Step': StepLRScheduler,
    'MultiStep': MultiStepLRScheduler,
    'Exponential': ExponentialLRScheduler,
    'CosineAnnealing': CosineAnnealingLRScheduler
}

def build_lr_scheduler(optimizer, cfg):
    type = cfg.TRAIN.LR_SCHEDULER.TYPE
    return LRs[type](optimizer, **cfg.TRAIN.LR_SCHEDULER.KWARGS)