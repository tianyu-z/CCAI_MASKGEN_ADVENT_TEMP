from comet_ml import Experiment
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
from pathlib import Path
import numpy as np
import yaml
import torch

from advent.model.deeplabv2 import get_deeplab_v2
from train_save_scripts import train_preview

from advent.utils.datasets import get_loader

from advent.utils.tools import (
    load_opts,
    set_mode,
    # avg_duration,
    flatten_opts,
    print_opts
)
# from time import time

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default="shared/advent.yml",
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument(
        "-d",
        "--data",
        help="yaml file for the data",
        default="shared/config.yml",
    )
    parser.add_argument(
        "-n",
        "--no_check",
        action="store_true",
        default=False,
        help="Prevent sample existence checking for faster dev",
    )
    return parser.parse_args()


def main():
    # --------------------------
    # -----  Load Options  -----
    # --------------------------
    args = get_arguments()
    print('Called with args:')
    print(args)
    assert args.cfg is not None, 'Missing cfg file'

    root = Path(__file__).parent.resolve()
    cfg = load_opts(path=root / args.cfg, default="shared/config.yml")
    cfg = set_mode("train", cfg)
    flats = flatten_opts(cfg)
    print_opts(flats)

    comet_exp = Experiment(workspace=cfg.workspace, project_name=cfg.project_name)
    flats = flatten_opts(cfg)
    comet_exp.log_parameters(flats)

    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')

    source_loader = get_loader(cfg, real=False, no_check=args.no_check)
    target_loader = get_loader(cfg, real=True, no_check=args.no_check)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
    
    train_preview(model, source_loader, target_loader, cfg, comet_exp)


if __name__ == '__main__':
    main()
