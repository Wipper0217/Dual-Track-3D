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
import numpy as np
from contrastive_learning.moco.loader import TwoCropsTransform
from contrastive_learning.moco.builder_moco import MoCo3D
from contrastive_learning.encoder.resnet.resnet import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from contrastive_learning.moco.utils import AverageMeter, ProgressMeter
from contrastive_learning.dataset.moco_dataset import MoCoDataset3D

from core.config_moco import cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('moco train')

parser = argparse.ArgumentParser(description='moco train')
parser.add_argument('--cfg', type=str, default='/root/data-tmp/core-track/experiments/moco/config_moco.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123568,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        num_workers=8,
        pin_memory=False,
    )

    return train_loader

def main_worker(cfg):
    base_encoder = resnet34
    model = MoCo3D(
        base_encoder=base_encoder,
        input_dim=512,
        mlp_dim=512,
        output_dim=256,
        T=0.2,
        mlp=True
    )
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), cfg.TRAIN.LR.LR,
    #                             momentum=cfg.TRAIN.MOMENTUM,
    #                             weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  betas=(0.9, 0.999),  # beta1 and beta2 parameters of the Adam optimizer, default (0.9, 0.999)
                                  eps=1e-8,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    train_loader = build_dataloader()
    save_per_epoch = 5
    logger.info("-----------------start training moco resnet34-----------------")
    for epoch in range(cfg.TRAIN.EPOCH):
        adjust_learning_rate(optimizer, epoch, cfg)
        train(train_loader, model, criterion, optimizer, epoch)
        save_model_name = "moco_encoder_q_{}_e{}.pth".format("resnet34", epoch + 1)
        if epoch + 1 > cfg.TRAIN.SAVE_EPOCH and (epoch+1) % save_per_epoch == 0:
            torch.save(model.encoder_q.state_dict(), cfg.TRAIN.SNAPSHOT_DIR+save_model_name)
        

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.to(device).train()

    end = time.time()
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        im_q = images[0].as_tensor().to(device)
        im_k = images[1].as_tensor().to(device)

        # compute output
        output, target = model(im_q=im_q, im_k=im_k)
        loss = criterion(output, target)
        losses.update(loss.item(), im_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.TRAIN.PRINT_FREQ == 0:
            progress.display(i)
        

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
    seed_torch(42)
    main()
    time.sleep(1000)
    #os.system("/usr/bin/shutdown")