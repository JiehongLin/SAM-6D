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
import pickle as cPickle
import json
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))


detetion_paths = {
    'ycbv': '../Instance_Segmentation_Model/log/sam/result_ycbv.json',
    'tudl': '../Instance_Segmentation_Model/log/sam/result_tudl.json',
    'tless': '../Instance_Segmentation_Model/log/sam/result_tless.json',
    'lmo': '../Instance_Segmentation_Model/log/sam/result_lmo.json',
    'itodd': '../Instance_Segmentation_Model/log/sam/result_itodd.json',
    'icbin': '../Instance_Segmentation_Model/log/sam/result_icbin.json',
    'hb': '../Instance_Segmentation_Model/log/sam/result_hb.json'
}


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
    parser.add_argument("--dataset",
                        type=str,
                        default="all",
                        help="")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        default="none",
                        help="path to checkpoint file")
    parser.add_argument("--iter",
                        type=int,
                        default=0,
                        help="iter num. for testing")
    parser.add_argument("--view",
                        type=int,
                        default=-1,
                        help="view number of templates")
    parser.add_argument("--exp_id",
                        type=int,
                        default=0,
                        help="experiment id")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus = args.gpus
    cfg.model_name = args.model
    cfg.log_dir = log_dir
    cfg.checkpoint_path = args.checkpoint_path
    cfg.test_iter = args.iter
    cfg.dataset = args.dataset

    if args.view != -1:
        cfg.test_dataset.n_template_view = args.view

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    return cfg



def test(model, cfg, save_path, dataset_name, detetion_path):
    model.eval()
    bs = cfg.test_dataloader.bs

    # build dataloader
    dataset = importlib.import_module(cfg.test_dataset.name)
    dataset = dataset.BOPTestset(cfg.test_dataset, dataset_name, detetion_path)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=cfg.test_dataloader.num_workers,
            shuffle=cfg.test_dataloader.shuffle,
            sampler=None,
            drop_last=cfg.test_dataloader.drop_last,
            pin_memory=cfg.test_dataloader.pin_memory
        )

    # prepare for target objects
    all_tem, all_tem_pts, all_tem_choose = dataset.get_templates()
    with torch.no_grad():
        dense_po, dense_fo = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    lines = []
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            torch.cuda.synchronize()
            end = time.time()

            for key in data:
                data[key] = data[key].cuda()
            n_instance = data['pts'].size(1)
            n_batch = int(np.ceil(n_instance/bs))

            pred_Rs = []
            pred_Ts = []
            pred_scores = []
            for j in range(n_batch):
                start_idx = j * bs
                end_idx = n_instance if j == n_batch-1 else (j+1) * bs
                obj = data['obj'][0][start_idx:end_idx].reshape(-1)

                # process inputs
                inputs = {}
                inputs['pts'] = data['pts'][0][start_idx:end_idx].contiguous()
                inputs['rgb'] = data['rgb'][0][start_idx:end_idx].contiguous()
                inputs['rgb_choose'] = data['rgb_choose'][0][start_idx:end_idx].contiguous()
                inputs['model'] = data['model'][0][start_idx:end_idx].contiguous()
                inputs['dense_po'] = dense_po[obj].contiguous()
                inputs['dense_fo'] = dense_fo[obj].contiguous()

                # make predictions
                with torch.no_grad():
                    end_points = model(inputs)
                pred_Rs.append(end_points['pred_R'])
                pred_Ts.append(end_points['pred_t'])
                pred_scores.append(end_points['pred_pose_score'])

            pred_Rs = torch.cat(pred_Rs, dim=0).reshape(-1, 9).detach().cpu().numpy()
            pred_Ts = torch.cat(pred_Ts, dim=0).detach().cpu().numpy() * 1000
            pred_scores = torch.cat(pred_scores, dim=0) * data['score'][0,:,0]
            pred_scores = pred_scores.detach().cpu().numpy()
            image_time = time.time() - end

            # write results
            scene_id = data['scene_id'].item()
            img_id = data['img_id'].item()
            image_time += data['seg_time'].item()
            for k in range(n_instance):
                line = ','.join((
                    str(scene_id),
                    str(img_id),
                    str(data['obj_id'][0][k].item()),
                    str(pred_scores[k]),
                    ' '.join((str(v) for v in pred_Rs[k])),
                    ' '.join((str(v) for v in pred_Ts[k])),
                    f'{image_time}\n',
                ))
                lines.append(line)


            t.set_description(
                "Test [{}/{}]".format(i+1, len(dataloder))
            )
            t.update(1)

    with open(save_path, 'w+') as f:
        f.writelines(lines)





if __name__ == "__main__":
    cfg = init()

    print("************************ Start Logging ************************")
    print(cfg)
    print("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # model
    print("creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()
    if cfg.checkpoint_path == 'none':
        checkpoint = os.path.join(cfg.log_dir, 'checkpoint_iter' + str(cfg.test_iter).zfill(6) + '.pth')
    else:
        checkpoint = cfg.checkpoint_path
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)


    if cfg.dataset == 'all':
        datasets = ['ycbv', 'tudl',  'lmo', 'icbin', 'tless', 'itodd' , 'hb']
        for dataset_name in datasets:
            print('begining evaluation on {} ...'.format(dataset_name))

            save_path = os.path.join(cfg.log_dir, dataset_name + '_eval_iter' + str(cfg.test_iter).zfill(6))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path,'result_' + dataset_name +'.csv')
            test(model, cfg, save_path, dataset_name, detetion_paths[dataset_name])

            print('saving to {} ...'.format(save_path))
            print('finishing evaluation on {} ...'.format(dataset_name))

    else:
        dataset_name = cfg.dataset
        print('begining evaluation on {} ...'.format(dataset_name))

        save_path = os.path.join(cfg.log_dir, dataset_name + '_eval_iter' + str(cfg.test_iter).zfill(6))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path,'result_' + dataset_name +'.csv')
        test(model, cfg,  save_path, dataset_name, detetion_paths[dataset_name])

        print('saving to {} ...'.format(save_path))
        print('finishing evaluation on {} ...'.format(dataset_name))





