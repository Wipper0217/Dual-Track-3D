import argparse
import builtins
import logging
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from contrastive_learning.moco.loader import TwoCropsTransform
from contrastive_learning.moco.builder_moco import MoCo3D
from contrastive_learning.encoder.resnet.resnet import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from contrastive_learning.moco.utils import AverageMeter, ProgressMeter
from contrastive_learning.dataset.moco_dataset import MoCoDataset3D
from contrastive_learning.moco.build_multi_branch_model import DBSST
from monai.transforms import (
    Compose, RandSpatialCropd, RandFlipd, RandAffined,
    RandGaussianSmoothd, RandGaussianNoised, RandAdjustContrastd,
    RandShiftIntensityd, ScaleIntensityd, EnsureTyped
)
from core.config_moco import cfg
from model.model_3d.resnet.resnet_ae_3d import ResNetAutoEncoder3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('moco train')

parser = argparse.ArgumentParser(description='moco train')
parser.add_argument('--cfg', type=str, default=r'D:\pycharmProject\pythonProject1\coding\3dTracking\experiments\moco\config_moco.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123568,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()
def build_dataloader():
    dataset = MoCoDataset3D(
        cfg.DATASET.ROOT,
        cfg.DATASET.NORMALIZE,
        cfg.DATASET.IMG_SIZE
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader

def main_worker(cfg):

    ae = ResNetAutoEncoder3D()

    base_encoder = resnet18
    moco = MoCo3D(
        base_encoder=base_encoder,
        input_dim=512,
        mlp_dim=512,
        output_dim=256,
        mlp=True
    )

    model = DBSST(
        autoencoder=ae,
        moco=moco
    )



    criterion = nn.CrossEntropyLoss().to(device)

    criterions = {
        'autoencoder': nn.MSELoss().to(device),
        'encoder': nn.CrossEntropyLoss().to(device)
    }

    # optimizer = torch.optim.SGD(model.parameters(), cfg.TRAIN.LR.LR,
    #                             momentum=cfg.TRAIN.MOMENTUM,
    #                             weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    optimizer1 = torch.optim.AdamW(model.parameters(),
                                  betas=(0.9, 0.999),  # beta1 and beta2 parameters of the Adam optimizer, default (0.9, 0.999)
                                  eps=1e-8,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer2 = torch.optim.AdamW(model.parameters(),
                                   betas=(0.9, 0.999),  # beta1 and beta2 parameters of the Adam optimizer, default (0.9, 0.999)
                                   eps=1e-8,
                                   weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizers = {
        'autoencoder': optimizer1,
        'encoder': optimizer2
    }

    train_loader = build_dataloader()
    for epoch in range(cfg.TRAIN.EPOCH):
        adjust_learning_rate(optimizer1, epoch, cfg)
        adjust_learning_rate(optimizer2, epoch, cfg)
        train(train_loader, model, criterions, optimizers, epoch)



def train(train_loader, model, criterions, optimizers, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ae_losses = AverageMeter('AutoEncoder Loss', ':.4e')
    moco_losses = AverageMeter('MoCo Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ae_losses, moco_losses],
        prefix="Epoch: [{}]".format(epoch))

    model.to(device).train()

    end = time.time()
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        im_q = images[0].as_tensor().to(device)
        im_k = images[1].as_tensor().to(device)

        # im_q = images[0].to(device)
        # im_k = images[1].to(device)

        # compute output
        outputs = model(im_q=im_q, im_k=im_k)
        # autoencoder branch
        output_ae = outputs['ae']
        criterion_ae = criterions['autoencoder']
        loss_ae = criterion_ae(output_ae, im_q)
        optimizer_ae = optimizers['autoencoder']
        optimizer_ae.zero_grad()
        loss_ae.backward()
        optimizer_ae.step()
        # encoder branch
        output_moco = outputs['moco']
        output, target = output_moco
        criterion_moco = criterions['encoder']
        loss_moco = criterion_moco(output, target)
        optimizer_moco = optimizers['encoder']
        optimizer_moco.zero_grad()
        loss_moco.backward()
        logger.info(f"autoencoder loss: {loss_ae.item()}, moco loss: {loss_moco.item()}")
        ae_losses.update(loss_ae.item(), im_q.size(0))
        moco_losses.update(loss_moco.item(), im_q.size(0))



        if i % 10 == 0:  # print every 10 batches
            print(f"Batch {i} gradients:")
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    print(f"{name}: {param_norm.item():.4f}")
            total_norm = total_norm ** (1. / 2)
            print(f"Total gradient norm: {total_norm:.4f}")
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.TRAIN.PRINT_FREQ == 0:
            progress.display(i)
        # save_model_name = "moco_{}.pth".format(epoch + 1)
        # # if epoch + 1 > cfg.TRAIN.SAVE_EPOCH:
        # #
        # #     torch.save(model.state_dict(), cfg.TRAIN.SNAPSHOT_DIR+save_model_name)
        # print("epoch {} save model: {}".format(epoch+1, save_model_name))

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.TRAIN.LR.LR
    if cfg.TRAIN.LR.COS:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / cfg.TRAIN.EPOCH))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    torch.cuda.empty_cache()
    cfg.merge_from_file(args.cfg)
    main_worker(cfg)

if __name__ == '__main__':
    main()