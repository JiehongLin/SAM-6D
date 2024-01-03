
import gorilla
from tqdm import tqdm
import argparse
import os
import sys
import os.path as osp
import time
import logging
import numpy as np
import random
import importlib

import torch
from torch.autograd import Variable
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

from solver import Solver, get_logger
from loss_utils import Loss

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="index of gpu")
    parser.add_argument("--model",
                        type=str,
                        default="pose_estimation_model",
                        help="name of model")
    parser.add_argument("--config",
                        type=str,
                        default="config/base.yaml",
                        help="path to config file")
    parser.add_argument("--exp_id",
                        type=int,
                        default=0,
                        help="experiment id")
    parser.add_argument("--checkpoint_iter",
                        type=int,
                        default=-1,
                        help="iter num. of checkpoint")
    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus = args.gpus
    cfg.model_name = args.model
    cfg.log_dir = log_dir
    cfg.checkpoint_iter = args.checkpoint_iter

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/training_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    if hasattr(cfg, 'pretrain_dir') and cfg.pretrain_dir is not None:
        logger.info('loading pretrained backbone from {}'.format(cfg.pretrain_dir))
        key1, key2 = model.load_state_dict(torch.load(cfg.pretrain_dir)['model'], strict=False)
    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()

    loss = Loss().cuda()
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))

    # dataloader
    batchsize = cfg.train_dataloader.bs
    num_epoch = cfg.training_epoch

    if cfg.lr_scheduler.type == 'WarmupCosineLR':
        num_iter = cfg.lr_scheduler.max_iters
        if hasattr(cfg, 'warmup_iter') and cfg.warmup_iter >0:
            num_iter = num_iter + cfg.warmup_iter
        iters_per_epoch = int(np.floor(num_iter / num_epoch))
    elif cfg.lr_scheduler.type == 'CyclicLR':
        iters_per_epoch = cfg.lr_scheduler.step_size_up+cfg.lr_scheduler.step_size_down
    train_dataset = importlib.import_module(cfg.train_dataset.name)
    train_dataset = train_dataset.Dataset(cfg.train_dataset, iters_per_epoch*batchsize)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.bs,
        num_workers=cfg.train_dataloader.num_workers,
        shuffle=cfg.train_dataloader.shuffle,
        sampler=None,
        drop_last=cfg.train_dataloader.drop_last,
        pin_memory=cfg.train_dataloader.pin_memory,
    )

    dataloaders = {
        "train": train_dataloader,
    }

    # solver
    Trainer = Solver(model=model, loss=loss,
                    dataloaders=dataloaders,
                    logger=logger,
                    cfg=cfg)
    Trainer.solve()

    logger.info('\nFinish!\n')
