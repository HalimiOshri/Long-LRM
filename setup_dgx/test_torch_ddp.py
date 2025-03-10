import os
import shutil
import copy
import argparse
from easydict import EasyDict as edict
import wandb
import yaml
import random
import time
import datetime
import numpy as np
import cv2
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision

from utils import create_logger, create_optimizer, create_scheduler, auto_resume_helper
from data.dataset import Dataset
from model.llrm import LongLRM

# torch and DDP setup
# rank = int(os.environ["RANK"])
# world_size = int(os.environ['WORLD_SIZE'])
# local_rank = int(os.environ['LOCAL_RANK'])
# local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
# group_rank = int(os.environ['GROUP_RANK'])
# device = "cuda:{}".format(local_rank)
# torch.cuda.set_device(device)
# torch.cuda.empty_cache()
# seed = 1111 + rank
# torch.manual_seed(seed)
# torch.random.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.distributed.init_process_group(backend='nccl')
# torch.distributed.barrier()
#
# torch.backends.cuda.matmul.allow_tf32 = config.use_tf32
# torch.backends.cudnn.allow_tf32 = config.use_tf32
#
# logger_dir = os.path.join(checkpoint_dir, 'logs')
# os.makedirs(logger_dir, exist_ok=True)
# logger = create_logger(output_dir=logger_dir, dist_rank=rank, name=config_name)
# logger.info(f"Rank {rank} / {world_size} with local rank {local_rank} / {local_world_size} and group rank {group_rank}")
# logger.info("Config:\n"+yaml.dump(config_dict, sort_keys=False))
#
# torch.distributed.barrier()