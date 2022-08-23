
# %%
import yaml
from datasets import MVTecDataset, MVTecDatasetDivided, MVTecDatasetScores
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from attacks.FGSM import FGSM
from attacks.PGD import PGD
from attacks.Gaussian import Gaussian
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# import faiss
import torch.nn.functional as F
from PIL import ImageFilter
import random
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import pandas as pd
import os


## Load Config ----------------------------------------------------------------
attack_config = None
with open('./attack_sde_config.yaml') as f:
    attack_config = yaml.load(f, Loader=yaml.FullLoader)

## Create Directories if does not exists -------------------------------------
if not os.path.exists(attack_config['results_path']):
    os.makedirs(attack_config['results_path'])

## Load SDE Scores -----------------------------------------------------------
sde_scores = np.load(attack_config['sde_scores'])
sde_labels = np.load(attack_config['sde_labels'])


min_score = np.min(sde_scores)
max_score = np.max(sde_scores)
normal_sde_scores = (sde_scores - min_score) / (max_score - min_score)
normal_sde_scores -= 0.5

## Load ResNet Model --------------------------------------------------------

import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone == 152:
            self.backbone = models.resnet152(pretrained=True).to(device)
        else:
            self.backbone = models.resnet18(pretrained=True).to(device)
        self.backbone.fc = torch.nn.Sequential(
             torch.nn.Linear(2048, 1),
        )
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        return z1

def freeze_parameters(model, backbone, train_fc=True):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False


simulated_model = Model(152).to(device)


## Load SDE Model --------------------------------------------------------------------

from ml_collections import config_dict
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
import ml_collections
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
from utils import save_checkpoint, restore_checkpoint
import yaml
from sampling import get_pc_sampler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import likelihood
import os

config = None
with open(attack_config['config_path']) as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader) 

config = config_dict.ConfigDict(config)

ckpt_filename = attack_config['ckpt_path']
 
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sampling_eps = 1e-5

batch_size = 64
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0 
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                              decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
            model=score_model, ema=ema)

config.device = device

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())
initial_step = int(state['step'])
print("initial_step", initial_step)


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


normal_class =  attack_config['normal_class']


## Test SDE labels and SDE Scores -------------------------------------------------------------

resnet_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


testset =  MVTecDatasetScores(attack_config['mvtec_root'], normal_class, resnet_transform, train=False, scores=normal_sde_scores)
testloader = torch.utils.data.DataLoader(testset, batch_size=attack_config['batch_size'], shuffle=False, num_workers=2)


dataset_labels = []
for i, data in enumerate(testloader):
  dataset_labels.extend(data[1].detach().numpy().tolist())


are_labels_correct = np.all(np.array(dataset_labels) == sde_labels)

if not are_labels_correct:
    raise ValueError('loaded SDE labels do not match Dataset labels')
else:
    print(f'Labels Matched!')

auc_score = roc_auc_score(sde_labels, normal_sde_scores)
print(f"AUC of laoded SDE scores: {auc_score * 100}")


## Train ResNet --------------------------------------------------------------

params_to_update = []

for p in simulated_model.parameters():
    if p.requires_grad:
        params_to_update.append(p)


learning_rate = attack_config['learning_rate'] #@param {type:"number"}

criterion = nn.MSELoss()
optimizer = optim.Adam(params_to_update, lr=learning_rate)


NUMBER_OF_EPOCHS =  attack_config['number_of_epochs']


simulated_model.train()
dataloader = torch.utils.data.DataLoader(testset, batch_size=attack_config['batch_size'], shuffle=True, num_workers=2)

auc_every_epoch = attack_config['auc_every_epoch']

for epoch in range(NUMBER_OF_EPOCHS):  # loop over the dataset multiple times

    simulated_model.eval()
    scores_simulated = []
    scores_true = []
    labels_true = []
    if auc_every_epoch:
      with torch.no_grad():
          for i, data in enumerate(dataloader):
              # get the inputs; data is a list of [inputs, labels]
              inputs, labels, scores = data
              inputs = inputs.float()
              
              inputs = inputs.to(device)
              labels = labels.to(device)

              labels_true.extend(labels.detach().cpu().numpy().tolist())


              outputs_simulated = simulated_model(inputs)

              scores_simulated.extend(outputs_simulated.reshape(-1).detach().cpu().numpy().tolist())


      print(f'AUC Simulated: {roc_auc_score(labels_true, scores_simulated) * 100}')

    running_loss = 0.0
    
    with tqdm(dataloader, unit="batch") as tepoch:
        simulated_model.train()
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, scores = data
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = scores.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs_simulated = simulated_model(inputs).float()
            loss = criterion(outputs_simulated, scores.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            tepoch.set_postfix({'Average Loss' : running_loss/(i+1) })




def do_attack(attack, simulated_model,  dataloader):

    adversarial_scores = []
    simulated_scores = []
    simulated_adversarial_scores = []
    all_labels = []

    for i, data in enumerate(dataloader):
        inputs, labels = data
        all_labels.extend(labels.numpy().flatten().tolist())

        inputs = inputs.to(device)

        simulated_output = simulated_model(transforms.Resize((224, 224))(inputs))
        print('Simulated Model Scores Before:')
        print(simulated_output.detach().cpu().numpy().flatten())

        adv_images = attack(transforms.Resize((224, 224))(inputs), labels)
        adversarial_simulated_output = simulated_model(adv_images)
        print('Simulated Model Scores After:')
        print(adversarial_simulated_output.detach().cpu().numpy().flatten())

        inputs_sde = transforms.Resize((config.data.image_size, config.data.image_size))(adv_images)

        if attack_config['uniform_dequantization']:
            inputs_sde = (torch.rand_like(inputs_sde, device=device) + inputs_sde * 255.) / 256. 

        img = scaler(inputs_sde)
        bpd, z, nfe = likelihood_fn(score_model, img)
        
        
        print('Scores BPD After:')
        print(bpd.detach().cpu().numpy().flatten())
        adversarial_scores.extend(bpd.detach().cpu().numpy().flatten().tolist())
        
        simulated_scores.extend(simulated_output.detach().cpu().numpy().flatten().tolist())
        simulated_adversarial_scores.extend(adversarial_simulated_output.detach().cpu().numpy().flatten().tolist())

    return adversarial_scores, simulated_scores, simulated_adversarial_scores, all_labels


def attack_model(attack, simulated_model, normal_loader, anomaly_loader):

    simulated_model.eval()

    result_normal = do_attack(attack=attack, simulated_model=simulated_model, dataloader=normal_loader)
    adversarial_scores_normal,\
    simulated_scores_normal,\
    simulated_adversarial_scores_normal,\
    labels_normal = result_normal

    result_anomaly = do_attack(attack=attack, simulated_model=simulated_model, dataloader=anomaly_loader)
    adversarial_scores_anomaly,\
    simulated_scores_anomaly,\
    simulated_adversarial_scores_anomaly,\
    labels_anomaly = result_anomaly

    adversarial_scores = adversarial_scores_normal + adversarial_scores_anomaly
    simulated_scores = simulated_scores_normal + simulated_scores_anomaly
    simulated_adversarial_scores = simulated_adversarial_scores_normal + simulated_adversarial_scores_anomaly
    all_labels = labels_normal + labels_anomaly

    return adversarial_scores, simulated_scores, simulated_adversarial_scores, all_labels




for p in simulated_model.parameters():
    p.requires_grad = True
    
normal_set = MVTecDatasetDivided(attack_config['mvtec_root'], normal_class, orig_transform, train=False, normal=True)
anomaly_set = MVTecDatasetDivided(attack_config['mvtec_root'], normal_class, orig_transform, train=False, normal=False)


normal_loader = torch.utils.data.DataLoader(normal_set, batch_size=4, shuffle=False, num_workers=2)
anomaly_loader = torch.utils.data.DataLoader(anomaly_set, batch_size=4, shuffle=False, num_workers=2)

results = {}
try:
    results = pd.read_csv(os.path.join(attack_config['results_path'], f'{attack_config["output_file_name"]}.csv')).to_dict(orient='list')
except:
    results['attack_config'] = []
    results['label'] = []
    results['auc_adv'] = []
    results['sim_adv'] = []
    results['auc_org'] = []
    results['sim_org'] = []


for attack in attack_config['attacks']:
    results['label'].append(normal_class)
    attack_type, attack_params = tuple(attack.items())[0]

    results['attack_config'].append(f'{attack_type} - {attack_params}')
    print(f'Runing {attack_type} Attack with config {attack_params}')
    
    attack_params['model'] = simulated_model

    attack_module = None
    if attack_type == 'PGD':
        attack_module = PGD(**attack_params)
    elif attack_type == 'FGSM':
        attack_module = FGSM(**attack_params)
    elif attack_type == 'Gaussian':
        attack_module = Gaussian(**attack_params)
    
    adversarial_scores,\
    simulated_scores,\
    simulated_adversarial_scores,\
    all_labels = \
    attack_model(attack_module, simulated_model, normal_loader, anomaly_loader)

    results['auc_adv'].append(roc_auc_score(all_labels, adversarial_scores) * 100)
    results['auc_org'].append(auc_score * 100)
    results['sim_adv'].append(roc_auc_score(all_labels, simulated_adversarial_scores) * 100)
    results['sim_org'].append(roc_auc_score(all_labels, simulated_scores) * 100)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(attack_config['results_path'], f'{attack_config["output_file_name"]}.csv'), index=False)
    print(f'Updated resutls at {os.path.join(attack_config["results_path"])}')