# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import math

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

from core.config import cfg


class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr
                for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__,
                                             self.lr_spaces)


class LogScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(math.log10(start_lr),
                                     math.log10(end_lr),
                                     epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 step=10, mult=0.1, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class MultiStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 steps=[10, 20, 30, 40], mult=0.1, epochs=50,
                 last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (len(steps)))
            else:
                mult = math.pow(end_lr/start_lr, 1. / len(steps))
        self.start_lr = start_lr
        self.lr_spaces = self._build_lr(start_lr, steps, mult, epochs)
        self.mult = mult
        self.steps = steps

        super(MultiStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, steps, mult, epochs):
        lr = [0] * epochs
        lr[0] = start_lr
        for i in range(1, epochs):
            lr[i] = lr[i-1]
            if i in steps:
                lr[i] *= mult
                print(lr[i])
        return np.array(lr, dtype=np.float32)


class LinearStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = np.linspace(start_lr, end_lr, epochs)
        super(LinearStepScheduler, self).__init__(optimizer, last_epoch)


class CosStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = self._build_lr(start_lr, end_lr, epochs)

        super(CosStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, end_lr, epochs):
        index = np.arange(epochs).astype(np.float32)
        lr = end_lr + (start_lr - end_lr) * \
            (1. + np.cos(index * np.pi / epochs)) * 0.5
        return lr.astype(np.float32)


class WarmUPScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        warmup = warmup.lr_spaces  # [::-1]
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)

class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, epochs, cycle_mult=1.0, start_lr=0.001, end_lr=0.00001, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.epochs = epochs
        self.cycle_mult = cycle_mult
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.cur_cycle = 0
        self.cur_cycle_steps = first_cycle_steps
        self.cur_step = 0
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def _build_lr(self):
        cycle_nums = self.epochs // self.first_cycle_steps


    # def get_lr(self):
    #     if self.cur_step == 0:
    #         self.cur_cycle += 1
    #         self.cur_cycle_steps = int(self.first_cycle_steps * (self.cycle_mult ** (self.cur_cycle - 1)))
    #         self.base_lrs = [base_lr + (max_lr - base_lr) * (self.cur_cycle / (self.cur_cycle + 1)) for base_lr, max_lr
    #                          in zip(self.base_lrs, [self.max_lr] * len(self.base_lrs))]
    #
    #     lr_lambda = 1 + (math.cos(math.pi * self.cur_step / self.cur_cycle_steps) - 1) / 2
    #     lrs = [self.base_lrs[i] + (self.max_lr - self.base_lrs[i]) * lr_lambda for i in range(len(self.base_lrs))]
    #
    #     self.cur_step += 1
    #     return lrs


LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler}



def _build_lr_scheduler(optimizer, config, epochs=50, last_epoch=-1):
    return LRs[config.TYPE](optimizer, last_epoch=last_epoch,
                            epochs=epochs, **config.KWARGS)


def _build_warm_up_scheduler(optimizer, epochs=50, last_epoch=-1):
    warmup_epoch = cfg.TRAIN.LR_WARMUP.EPOCH
    sc1 = _build_lr_scheduler(optimizer, cfg.TRAIN.LR_WARMUP,
                              warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, cfg.TRAIN.LR,
                              epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, epochs=50, last_epoch=-1):
    if cfg.TRAIN.LR_WARMUP.WARMUP:
        return _build_warm_up_scheduler(optimizer, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, cfg.TRAIN.LR,
                                   epochs, last_epoch)


if __name__ == '__main__':
    import torch.nn as nn
    from torch.optim import SGD
    from pysot.core.config import cfg
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3)
    trainable_params = [{'params': Net().parameters(),
                         'lr': cfg.TRAIN.LR}]
    start_lr = 0.0001
    end_lr = 0.001
    optimizer = SGD(Net().parameters(), lr=start_lr)
    epochs = 20
    # warm_up_epoch = 10
    # log = LogScheduler(optimizer=optimizer, start_lr=start_lr, end_lr=end_lr, epochs=epochs)
    # cosine = CosStepScheduler(optimizer=optimizer, start_lr=start_lr, end_lr=end_lr, epochs=epochs)
    # linear = LinearStepScheduler(optimizer=optimizer, start_lr=start_lr, end_lr=end_lr, epochs=epochs)
    # step = StepScheduler(optimizer=optimizer, start_lr=start_lr, end_lr=end_lr)
    # warmup_linear = LinearStepScheduler(optimizer=optimizer, start_lr=end_lr, end_lr=start_lr, epochs=warm_up_epoch)
    # warmup_cosine = CosStepScheduler(optimizer=optimizer, start_lr=start_lr, end_lr=end_lr, epochs=(epochs-warm_up_epoch)//2)
    # warmup = WarmUPScheduler(optimizer=optimizer, warmup=warmup_linear, normal=warmup_cosine, epochs=epochs)
    multistep = MultiStepScheduler(optimizer=optimizer, start_lr=start_lr, epochs=epochs, steps=[10, 15])
    x = np.linspace(1, epochs, epochs, dtype=np.int64)
    # y1 = log.lr_spaces
    # y2 = cosine.lr_spaces
    # y3 = linear.lr_spaces
    # y4 = step.lr_spaces
    # y5 = warmup.lr_spaces
    y6 = multistep.lr_spaces
    import matplotlib.pyplot as plt
    # # plt.plot(x, y1, label='log')
    # # plt.plot(x, y2, label='cosine')
    # # plt.plot(x, y3, label='linear')
    # # plt.plot(x, y4, label='step')
    # # plt.plot(x, y5, label='warmup')
    plt.plot(x, y6, label='multi_step')
    plt.show()

