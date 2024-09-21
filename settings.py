import torch
import os
import numpy as np

""" 1. File directory Settings """
CSV_FILE_DIRC     = "EEG_csv"
EEG_FILE_DIRC     = "D:\EEG_Dataset"
VIS_FILE_DIRC     = "visualization"
MODEL_FILE_DIRC   = "Model"

os.makedirs(CSV_FILE_DIRC,   exist_ok = True) 
os.makedirs(VIS_FILE_DIRC,   exist_ok = True)
os.makedirs(MODEL_FILE_DIRC, exist_ok = True)


""" 2. Pre-processing Settings"""
NEW_FREQUENCY     = 128    # New frequency     
CHANNELS_NAME     = ['F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3', 'O1',
                     'Fz', 'Cz', 'Pz',
                     'O2', 'P4', 'C4', 'F4', 'Fp2', 'T6', 'T4', 'F8']

AVE_CHANNELS_NAME = ['F7-AVE', 'T3-AVE', 'T5-AVE', 'Fp1-AVE', 'F3-AVE', 'C3-AVE', 'P3-AVE', 'O1-AVE',
                     'Fz-AVE', 'Cz-AVE', 'Pz-AVE',
                     'O2-AVE', 'P4-AVE', 'C4-AVE', 'F4-AVE', 'Fp2-AVE', 'T6-AVE', 'T4-AVE', 'F8-AVE']

COL_DTYPE         = {chan:np.float32 for chan in AVE_CHANNELS_NAME}


DURATION          = 10          # Number of duration taken input and output
NORMALIZE_CONS_1  = 0.0012
NORMALIZE_CONS_2  = 1.18


""" 3. Diffusion Model - WaveGrad Settings """
config = torch.nn.Module()
config.n_mels = 32
#  1280 = 2^7 * 10 = 4 * 4 * 5 * 4 * 4
config.factors = [4, 4, 5, 4, 4]                # Factor to upsampling and downsampling the dataset[-1]
config.upsampling_preconv_out_channels = 768
config.upsampling_out_channels = [512, 512, 256, 128, 128]
config.upsampling_dilations = [[1, 2, 1, 2],
                               [1, 2, 1, 2],
                               [1, 2, 4, 8],
                               [1, 2, 4, 8],
                               [1, 2, 4, 8]]
config.downsampling_preconv_out_channels = 32
config.downsampling_out_channels = [128, 128, 256, 512]
config.downsampling_dilations = [[1, 2, 4], 
                                 [1, 2, 4], 
                                 [1, 2, 4], 
                                 [1, 2, 4]]


""" 4. Training Settings """
n_steps               = 500        # Maximum number of steps used to diffuse the signal (larger the better in training)
MAX_COUNT             = 30         # Maximum count in decreasing of validation loss
NUM_EPOCHS            = 10000      # Number of epochs to train diffusion model

MAX_COUNT_F1_SCORE    = 10         # Maximum count in not increasing in validation f1 score
NUM_EPOCHS_CLASSIFIER = 101        # Number of epochs to train classifiers

LEARNING_RATE         = 0.001
BATCH_SIZE            = 32
device                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


""" 5. Sampling constant"""
def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-3, 3, n_timesteps)          # Original is -6 to 6
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

betas              = make_beta_schedule(schedule='linear', n_timesteps=n_steps, start=1e-4, end=1e-2)   # Original is linear
alphas             = 1 - betas
alphas_prod        = torch.cumprod(alphas, 0)                                      # Cumulative product of alphas
alphas_prod_p      = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_prod_p_sqrt = alphas_prod_p.sqrt()                                   # sqrt(alpha_bar)
alphas_bar_sqrt    = torch.sqrt(alphas_prod)               # Un-use variable

sqrt_recip_alphas_cumprod = (1 / alphas_prod).sqrt()
sqrt_alphas_cumprod_m1    = (1 - alphas_prod).sqrt() * sqrt_recip_alphas_cumprod 

posterior_mean_coef_1     = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
posterior_mean_coef_2     = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))

posterior_variance        = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)