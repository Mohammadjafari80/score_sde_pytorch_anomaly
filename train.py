from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
from utils import save_checkpoint, restore_checkpoint


import torch.nn as nn
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
from ml_collections import config_dict

#import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
# from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets


#!pip install ml_collections
import ml_collections
import torch

import yaml


train_config = None
with open('./train_config.yaml') as f:
    train_config = yaml.load(f, Loader=yaml.FullLoader)

print(train_config)

config = None
with open(train_config['config_path']) as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader) 

config = config_dict.ConfigDict(config)


import logging
workdir = train_config['work_dir']
tf.io.gfile.makedirs(workdir)
gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
handler = logging.StreamHandler(gfile_stream)
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel('INFO')

workdir = os.path.join(workdir, train_config['normal_class'])


import matplotlib.pyplot as plt
sample_dir = os.path.join(workdir, "samples")
tf.io.gfile.makedirs(sample_dir)

tb_dir = os.path.join(workdir, "tensorboard")
tf.io.gfile.makedirs(tb_dir)
###writer = SummaryWriter(tb_dir)

from models.utils import create_model
from losses import get_optimizer, optimization_manager, get_step_fn


# Initialize model.
score_model =  create_model(config)
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
optimizer =  get_optimizer(config, score_model.parameters())
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)


# Create checkpoints directory
checkpoint_dir = os.path.join(workdir, "checkpoints")
# Intermediate checkpoints to resume training after pre-emption in cloud environments
checkpoint_meta_dir = os.path.join(workdir,  "checkpoints-meta")
tf.io.gfile.makedirs(checkpoint_dir)
tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
# Resume training when intermediate checkpoints are detected
state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
initial_step = int(state['step'])
print("initial_step",initial_step)

from datasets import get_data_scaler, get_data_inverse_scaler
from sampling import get_sampling_fn

# Create data normalizer and its inverse
scaler =  get_data_scaler(config)
inverse_scaler =  get_data_inverse_scaler(config)

# Setup SDEs
if config.training.sde.lower() == 'vpsde':
    sde =  VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
elif config.training.sde.lower() == 'subvpsde':
    sde =  subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
elif config.training.sde.lower() == 'vesde':
    sde =  VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

# Build one-step training and evaluation functions
optimize_fn =  optimization_manager(config)
continuous = config.training.continuous
reduce_mean = config.training.reduce_mean
likelihood_weighting = config.training.likelihood_weighting
train_step_fn = get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
eval_step_fn = get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                reduce_mean=reduce_mean, continuous=continuous,
                                likelihood_weighting=likelihood_weighting)

# Building sampling functions
if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
    sampling_fn =   get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


num_train_steps = train_config['n_iters']
normal_class = train_config['normal_class']
mvtec_root = train_config['mvtec_root']
root_dir_train = os.path.join(mvtec_root, normal_class, 'train')


from torchvision.datasets import ImageFolder
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset 
import numpy as np
import time


orig_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor()
        ])

dataset_train_good = ImageFolder(root=root_dir_train, transform=orig_transform)

Mvtecad_train_loader = torch.utils.data.DataLoader(
        dataset_train_good,
        batch_size=train_config['batch_size'],
        shuffle=True,
    )


import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
from torchvision.utils import make_grid, save_image



sampling_shape = (9, config.data.num_channels,
                  config.data.image_size, config.data.image_size)

sampling_fn =   get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

total_loss=[]


def timer(start,end):
             hours, rem = divmod(end-start, 3600)
             minutes, seconds = divmod(rem, 60)
             print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))




for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    a_loss = 0
    tik = time.time()

    for myiter, (batch_images, batch_labels) in enumerate(Mvtecad_train_loader):
      batch = batch_images.cuda()  
      batch = scaler(batch)
      loss = train_step_fn(state, batch)
      total_loss.append(loss.item())
      a_loss += loss.item()
    

    tok=time.time()
    timer(tik,tok)
    print(f'epoch loss: {a_loss/myiter:.5f}, Step: {step:04d}')

    if train_config['save_checkpoints']:
        if step > 0 and step % train_config['save_checkpoints_every'] == 0 :
            save_checkpoint(f'/last_ckpt_{normal_class}.pth', state) 

    if step > 0 and step % train_config['sample_every'] == 0:
        if train_config['sample_due_training']:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
            this_sample_dir = os.path.join(sample_dir, f'{step:04d}')
            tf.io.gfile.makedirs(this_sample_dir)
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = make_grid(sample, nrow, padding=2)
            sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)

            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, f"sample-{step:04d}.np"), "wb") as fout:
                np.save(fout, sample)

            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, f"sample-{step:04d}.png"), "wb") as fout:
                save_image(image_grid, fout)

    tok=time.time()
    timer(tik,tok)



