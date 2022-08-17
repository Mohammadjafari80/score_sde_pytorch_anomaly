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
from tqdm import tqdm
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
from sampling import get_pc_sampler
from datasets import MVTecDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


test_config = None
with open('./test_config.yaml') as f:
    test_config = yaml.load(f, Loader=yaml.FullLoader)

print(test_config)


if not os.path.exists(test_config['train_save_path']):
    os.makedirs(test_config['train_save_path'])

if not os.path.exists(test_config['test_save_path']):
    os.makedirs(test_config['test_save_path'])

if not os.path.exists(test_config['auc_save_path']):
    os.makedirs(test_config['auc_save_path'])

config = None
with open(test_config['config_path']) as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader) 

config = config_dict.ConfigDict(config)

ckpt_filename = test_config['ckpt_path']
 
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sampling_eps = 1e-5

batch_size = 64
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 # param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                              decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
            model=score_model, ema=ema)


state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())
initial_step = int(state['step'])
print("initial_step",initial_step)


img_size = config.data.image_size
channels = config.data.num_channels
shape = (1, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor # param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector # param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 # param {"type": "number"}
n_steps =  1 # param {"type": "integer"}
probability_flow = False # param {"type": "boolean"}
sampling_fn =  get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)


likelihood_fn = likelihood.get_likelihood_fn(sde,                                              
                                             inverse_scaler,                                             
                                             eps=1e-5)


orig_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])


train_loader, test_loader = None, None
normal_class =  test_config['normal_class']


if test_config['dataset'] == 'mvtec':
    trainset = MVTecDataset(test_config['mvtec_root'], normal_class, orig_transform, train=True)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=test_config['batch_size'], num_workers=2)
    testset = MVTecDataset(test_config['mvtec_root'], normal_class, orig_transform, train=False)
    if test_config['quick_estimate']:
        sample_count = int(len(testset) * test_config['portion_of_sample'])
        testset, _ = torch.utils.data.random_split(testset, [sample_count, len(testset) - sample_count])
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=test_config['batch_size'], num_workers=2)  

elif test_config['dataset'] == 'cifar':
    cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    normal_class_indx = cifar_labels.index(normal_class)

    trainset = CIFAR10(root=os.path.join(test_config['cifar_root'], 'cifar10'), train=True, download=True, transform=orig_transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=test_config['batch_size'], shuffle=False, num_workers=2)

    testset = CIFAR10(root=os.path.join(test_config['cifar_root'], 'cifar10'), train=False, download=True, transform=orig_transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    if test_config['quick_estimate']:
        sample_count = int(len(testset) * test_config['portion_of_sample'])
        testset, _ = torch.utils.data.random_split(testset, [sample_count, len(testset) - sample_count])

    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_config['batch_size'], shuffle=False, num_workers=2)

    
    
# TEST SCORES & AUC on TEST SET ---- NO SHUFFLE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = test_config['dataset']

import pandas as pd

results = {}
try:
    results = pd.read_csv(os.path.join(test_config['auc_save_path'], f'auc-{dataset}-{normal_class}.csv')).to_dict(orient='list')
except:
    results['auc'] = []
    results['label'] = []
    results['repeat'] = []



all_bpds = []


for repeat in range(test_config['bpd_num_repeats']):
    print('-' * 50)
    print(f'Evaluating Scores on TestSet - {"Quick" if  test_config["quick_estimate"] else "Complete"} mode - Pass {repeat+1}/{test_config["bpd_num_repeats"]}')
    print('-' * 50)

    scores = []
    labels = []

    with tqdm(test_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f'Batch : {i+1}/{len(test_loader)} | Pass: {repeat+1}/{test_config["bpd_num_repeats"]}')
                new_batch = data[0].to(device)
                if test_config['uniform_dequantization']:
                    new_batch =  (torch.rand_like(new_batch, device=device) + new_batch * 255.) / 256. 
                img = scaler(new_batch)
                current_labels = data[1].detach().cpu().numpy()
                bpd, z, nfe = likelihood_fn(score_model, img)
                current_scores = bpd.detach().cpu().numpy().reshape(-1)
                scores.extend(current_scores.tolist())
                labels.extend(current_labels.tolist())
                tepoch.set_postfix({'BPDs' : current_scores, 'Labels' : current_labels})
                

    from sklearn.metrics import roc_auc_score

    all_bpds.append(scores)
    auc_score = roc_auc_score(labels, scores)
    results['auc'].append(auc_score * 100)
    results['label'].append(normal_class)
    results['repeat'].append(repeat)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(test_config['auc_save_path'], f'auc-{dataset}-{normal_class}.csv'), index=False)
    print(f'Added AUC resutls at {os.path.join(test_config["auc_save_path"])}')
    print(f'AUC Score - Test - Pass {repeat+1} is : {auc_score * 100}')


    if not test_config['quick_estimate']:

        with open(os.path.join(test_config['test_save_path'], f'score-{dataset}-{normal_class}-{repeat}-test.npy'), 'wb') as f:
                np.save(f, np.array(scores))

        with open(os.path.join(test_config['test_save_path'], f'labels-{dataset}-{normal_class}-{repeat}-test.npy'), 'wb') as f:
                np.save(f, np.array(labels))

with open(os.path.join(test_config['test_save_path'], f'scores-{dataset}-{normal_class}-all-test.npy'), 'wb') as f:
                np.save(f, np.array(all_bpds))


# TEST SCORES & AUC on TRAIN SET ---- NO SHUFFLE

scores = []

if test_config['test_on_train']:
    if not test_config['quick_estimate']:
        print('-' * 50)
        print('Evaluating Scores on TrainSet...')
        print('-' * 50)
        with tqdm(train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f'Batch : {i+1}/{len(train_loader)}')
                    new_batch = data[0].to(device)
                    if test_config['uniform_dequantization']:
                        new_batch = (torch.rand_like(new_batch, device=device) + new_batch * 255.) / 256. 
                    img = scaler(new_batch)
                    bpd, z, nfe = likelihood_fn(score_model, img)
                    current_scores = bpd.detach().cpu().numpy().reshape(-1)
                    scores.extend(current_scores.tolist())
                    tepoch.set_postfix({'BPDs' : current_scores})



        with open(os.path.join(test_config['train_save_path'], f'score-{normal_class}-train.npy'), 'wb') as f:
                np.save(f, np.array(scores))


