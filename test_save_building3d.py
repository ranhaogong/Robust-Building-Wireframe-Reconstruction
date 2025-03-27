import os
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import torch.distributed as dist
from dataset.data_utils import build_dataloader_Building3DDatasetOutput
from test_util import save_wireframe
from model.roofnet import RoofNet
from torch import optim
from utils import common_utils
from model import model_utils


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../GithubDeepRoof', help='dataset path')
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='1', help='gpu for training')
    parser.add_argument('--test_tag', type=str, default='pts6', help='extra tag for this experiment')

    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    return args, cfg


def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    extra_tag = args.test_tag
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    assert output_dir.exists(), '%s does not exist!!!' % str(output_dir)
    ckpt_dir = output_dir / 'ckpt'
    output_dir = output_dir / 'save'
    output_dir.mkdir(parents=True, exist_ok=True)

    # log_file = output_dir / 'log.txt'
    # logger = common_utils.create_logger(log_file)

    # logger.info('**********************Start logging**********************')
    # for key, val in vars(args).items():
    #     logger.info('{:16} {}'.format(key, val))
    # common_utils.log_config_to_file(cfg, logger=logger)

    test_loader = build_dataloader_Building3DDatasetOutput(args.data_path, args.batch_size, cfg.DATA, training=False, logger=None, color=cfg.COLOR, nir=cfg.NIR, intensity=cfg.INTENSITY, fpfh=getattr(cfg, 'FPFH', False), mrgd=getattr(cfg, 'MRGD', False))
    net = RoofNet(cfg.MODEL, color=cfg.COLOR, nir=cfg.NIR, intensity=cfg.INTENSITY, fpfh=getattr(cfg, 'FPFH', False), lovasz=getattr(cfg, 'LOVASZ', False), mrgd=getattr(cfg, 'MRGD', False))
    net.cuda()
    net.eval()
    print("ckpt_dir: ", ckpt_dir)
    ckpt_list = ['/data/haoran/Point2Roof/output/building3d_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_lovasz_edge_dbscan_003_cross_attention/ckpt/checkpoint_epoch_144.pth']
    # /data/haoran/Point2Roof/output/building3d_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_lovasz_wavelet_edge/ckpt/checkpoint_epoch_144.pth
    # ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    print("ckpt_list: ", ckpt_list)
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        model_utils.load_params(net, ckpt_list[-1], logger=None)
        print("pth: ", ckpt_list[-1])

    print('**********************Start saving**********************')
    # logger.info(net)

    save_wireframe(net, test_loader, output_dir)

    


if __name__ == '__main__':
    main()
