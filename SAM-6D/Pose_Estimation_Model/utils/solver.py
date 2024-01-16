import os
import time
import logging
from tqdm import tqdm
import pickle as cPickle

import torch
import torch.optim as optim

import gorilla
from gorilla.solver.build import build_optimizer, build_lr_scheduler
from tensorboardX import SummaryWriter


class Solver(gorilla.solver.BaseSolver):
    def __init__(self, model, loss, dataloaders, logger, cfg, coarse_model=None):
        super(Solver, self).__init__(
            model=model,
            dataloaders=dataloaders,
            cfg=cfg,
        )
        self.loss = loss
        self.logger = logger
        self.logger.propagate = 0

        self.coarse_model = coarse_model.eval() if coarse_model is not None else coarse_model

        tb_writer_ = tools_writer(
            dir_project=cfg.log_dir, num_counter=2, get_sum=False)
        tb_writer_.writer = self.tb_writer
        self.tb_writer = tb_writer_
        self.iters_to_print = cfg.iters_to_print

        if cfg.checkpoint_iter != -1:
            logger.info("=> loading checkpoint from iter {} ...".format(cfg.checkpoint_iter))
            checkpoint = os.path.join(cfg.log_dir, 'checkpoint_iter' + str(cfg.checkpoint_iter).zfill(6) + '.pth')
            checkpoint_file = gorilla.solver.resume(model=model, filename=checkpoint, optimizer=self.optimizer, scheduler=self.lr_scheduler)
            start_epoch = checkpoint_file['epoch']+1
            start_iter = checkpoint_file['iter']
            del checkpoint_file
        else:
            start_epoch = 1
            start_iter = 0
        self.epoch = start_epoch
        self.iter = start_iter

        if hasattr(cfg, 'warmup_iter') and cfg.warmup_iter > 0:
            self.warmup_optimizer = build_optimizer(model, cfg.warmup_optimizer)
            self.warmup_scheduler = build_lr_scheduler(self.warmup_optimizer, cfg.warmup_lr_scheduler)


    def solve(self):
        while self.epoch <= self.cfg.training_epoch:
            self.logger.info('\nEpoch {} :'.format(self.epoch))

            end = time.time()
            dict_info_train = self.train()
            train_time = time.time()-end

            dict_info = {'train_time(min)': train_time/60.0}
            for key, value in dict_info_train.items():
                if 'loss' in key:
                    dict_info['train_'+key] = value

            ckpt_path = os.path.join(
                self.cfg.log_dir, 'checkpoint_iter'+ str(self.iter).zfill(6) +'.pth')
            gorilla.solver.save_checkpoint(
                model=self.model, filename=ckpt_path, optimizer=self.optimizer, scheduler=self.lr_scheduler, meta={'iter': self.iter, "epoch": self.epoch})

            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            self.logger.warning(write_info)
            self.epoch += 1

    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()
        self.dataloaders["train"].dataset.reset()

        for i, data in enumerate(self.dataloaders["train"]):
            torch.cuda.synchronize()
            data_time = time.time()-end

            if hasattr(self.cfg, 'warmup_iter') and self.cfg.warmup_iter > 0:
                if self.iter >= self.cfg.warmup_iter:
                    optimizer = self.optimizer
                    lr_scheduler = self.lr_scheduler
                else:
                    optimizer = self.warmup_optimizer
                    lr_scheduler = self.warmup_scheduler
            else:
                optimizer = self.optimizer
                lr_scheduler = self.lr_scheduler


            optimizer.zero_grad()
            loss, dict_info_step = self.step(data, mode)
            dict_info_step['lr'] = lr_scheduler.get_last_lr()[0]
            forward_time = time.time()-end-data_time

            loss.backward()
            optimizer.step()
            backward_time = time.time() - end - forward_time-data_time

            dict_info_step.update({
                'T_data': data_time,
                'T_forward': forward_time,
                'T_back': backward_time,
            })
            self.log_buffer.update(dict_info_step)
            self.iter += 1

            if i % self.iters_to_print == 0:
                # ipdb.set_trace()
                self.log_buffer.average(self.iters_to_print)
                prefix = 'Iter {} Train - '.format(str(self.iter).zfill(6))
                write_info = self.get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            end = time.time()

            lr_scheduler.step()

        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def evaluate(self):
        mode = 'eval'
        self.model.eval()

        for i, data in enumerate(self.dataloaders["eval"]):
            with torch.no_grad():
                _, dict_info_step = self.step(data, mode)
                self.log_buffer.update(dict_info_step)
                if i % self.iters_to_print == 0:
                    self.log_buffer.average(self.iters_to_print)
                    prefix = '[{}/{}][{}/{}] Test - '.format(
                        self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["eval"]))
                    write_info = self.get_logger_info(
                        prefix, dict_info=self.log_buffer._output)
                    self.logger.info(write_info)
                    self.write_summary(self.log_buffer._output, mode)
        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def step(self, data, mode):
        torch.cuda.synchronize()
        for key in data:
            data[key] = data[key].cuda()
        if self.coarse_model is not None:
            with torch.no_grad():
                data = self.coarse_model(data)
        end_points = self.model(data)
        dict_info = self.loss(end_points)
        loss_all = dict_info['loss']

        for key in dict_info:
            dict_info[key] = float(dict_info[key].item())

        return loss_all, dict_info

    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            elif 'lr' in key:
                info = info + '{}: {:.6f}\t'.format(key, value)
            else:
                info = info + '{}: {:.5f}\t'.format(key, value)

        return info

    def write_summary(self, dict_info, mode):
        keys = list(dict_info.keys())
        values = list(dict_info.values())
        if mode == "train":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=0, prefix="train_")
        elif mode == "eval":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=1, prefix="eval_")
        else:
            assert False



class tools_writer():
    def __init__(self, dir_project, num_counter, get_sum):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        # writer = SummaryWriter(dir_project)
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = []
        for i in range(num_counter):
            self.list_couter.append(0)

    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix+name, float(value), self.list_couter[index_counter])

        self.list_couter[index_counter] += 1

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0


def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger