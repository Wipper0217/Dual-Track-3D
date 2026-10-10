import argparse
import logging
import os
import random
import time 

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset.dataset_3d_new import Dataset3DNew
from dataset.load_dataset import LoadDataSet
from matplotlib import pyplot as plt
from comparative_experiments.dcn.AutoEncoder_DCN import AutoEncoderV1
from dataset.crop_utils import get_crop_function_new
import numpy as np
from core.config_3d import cfg
from utils.my_lr_scheduler import build_lr_scheduler

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('using_trainset')
parser = argparse.ArgumentParser(description='3d tracking')
parser.add_argument('--cfg', type=str, default='/root/code/tracking/3dTracking/experiments/auto-encoder/config_new.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123568,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_dataloader():
    logger.info("build train dataset start: image size={}, random_range={}, normalize={}, method={}, in_channels={}, channels_24_direction={}".format(
        cfg.DATASET.IMG_SIZE,
        cfg.DATASET.RANDOM_RANGE,
        cfg.DATASET.NORMALIZE,
        cfg.DATASET.METHOD,
        cfg.DATASET.IN_CHANNELS,
        cfg.DATASET.CHANNELS_24_DIRECTION
    ))
    crop_function = get_crop_function_new(cfg.DATASET.METHOD, cfg.DATASET.IN_CHANNELS, cfg.DATASET.CHANNELS_24_DIRECTION)

    dataset = Dataset3DNew(
        cfg.DATASET.ROOT,
        cfg.DATASET.NORMALIZE,
        crop_function,
        cfg.DATASET.IMG_SIZE
    )

    logger.info("build train dataset done!------batch size: {}, crop function: {}".format(cfg.TRAIN.BATCH_SIZE, crop_function.__name__))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=False,
    )
    return dataloader

def build_opt_lr(model):

    trainable_params = []
    trainable_params += [
        {"params": model.encoder.parameters(),
         'lr': cfg.TRAIN.BASE_LR
         }
    ]

    trainable_params += [
        {"params": model.decoder.parameters(),
         'lr': cfg.TRAIN.BASE_LR
         }
    ]

    optimizer = torch.optim.AdamW(trainable_params,
                                  betas=(0.9, 0.999),  # beta1 and beta2 parameters of the Adam optimizer, default (0.9, 0.999)
                                  eps=1e-8,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    logger.info("lr scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))
    lr_scheduler = build_lr_scheduler(optimizer, cfg)
    return optimizer, lr_scheduler



def main():
    cfg.merge_from_file(args.cfg)
    # in_channels = 24
    # dataset = Dataset3D(
    #     "/root/data-fs/deshengl/clust/clust_trainset_3d",
    #     in_channels,
    # )
    # print(f"Dataset loaded. Total samples: {len(dataset)}")
    # dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    dataloader = build_dataloader()


    #model = ResNetAutoEncoder(in_channels=cfg.DATASET.IN_CHANNELS, normalize=cfg.DATASET.NORMALIZE, num_layers=18)
    model = AutoEncoderV1(normalize=cfg.DATASET.NORMALIZE, in_channels=cfg.DATASET.IN_CHANNELS)

    model = model.to(device).train()
    #print("batch size = {}".format(cfg.TRAIN.BATCH_SIZE))
    logger.info("Training model: {}, total epoch: {}".format(model.get_name(), cfg.TRAIN.EPOCH))
    optimizer, lr_scheduler = build_opt_lr(model)
    for epoch in range(cfg.TRAIN.EPOCH):
        logger.info("epoch: {}, lr_rate: {}".format(epoch+1, lr_scheduler.get_current_lr()))
        total_loss = 0.0
        for idx, data in enumerate(dataloader):
            data = data.to(device)
            # print("batch: {}, input={}".format(idx+1, data.shape))
            optimizer.zero_grad()
            output = model(data)
            #logger.debug(output[0, 0, 0, 0])
            # print("output: {}".format(output.shape))
            criterion = nn.MSELoss()
            loss = criterion(output, data)
            #print("epoch: {} batch: {} mse loss={}".format(epoch+1, idx+1, loss))

            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         logger.debug("name: {}, grad: {}".format(name, param.grad.abs().mean()))
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        lr_scheduler.step()
        info = "epoch: {} total loss={}, avg loss={}".format(epoch+1, total_loss, avg_loss)
        logger.info(info)
        #print(info)
        if cfg.DATASET.METHOD != '2d':
            save_model_name = "/{}_3d_{}.pth".format(model.get_name(), epoch+1)
        else:
            if cfg.DATASET.IN_CHANNELS != 24:
                save_model_name = "/dcn_{}_{}c_new_e{}.pth".format(model.get_name(), cfg.DATASET.IN_CHANNELS, epoch+1)
            else:
                save_model_name = "/dcn_{}_{}c_{}_{}.pth".format(model.get_name(), cfg.DATASET.IN_CHANNELS, cfg.DATASET.CHANNELS_24_DIRECTION, epoch+1)
        if epoch + 1 > cfg.TRAIN.SAVE_EPOCH:
            torch.save(model.state_dict(), cfg.TRAIN.SNAPSHOT_DIR+save_model_name)
            #print("epoch {} save model: {}".format(epoch+1, save_model_name))

if __name__ == '__main__':
    print(device)
    seed_torch()
    main()
    time.sleep(100)
    #os.system("/usr/bin/shutdown")
