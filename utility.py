import numpy as np
import pandas as pd
import mne
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from settings import *
import pywt
import json
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from ummc_db_util import Compumedics


def process_edf(eeg_file, num, SKIP_FIRST, SKIP_LAST, describe_it = False):
    print("For the file: ", eeg_file)
    
    # 1. Load the edf file, convert to pandas dataframe and drop some of the data 
    raw = Compumedics(eeg_file).export_to_mne_raw() # Return a raw object   
    df  = raw.to_data_frame() / 1e6      # Get dataframe (Scaling = *1e6, it convert V to µV)
    if describe_it:
        print(df.describe())
    print("Before drop some row, shape =", df.shape)
    
    ORIGINAL_FREQUENCY = raw.info["sfreq"]
    df = df.iloc[int(SKIP_FIRST*ORIGINAL_FREQUENCY):,:]
    df = df.iloc[:-int(SKIP_LAST*ORIGINAL_FREQUENCY),:]
    
    # 2. Remain only 19 EEG signal and arrange the order of signals
    print("After  drop some row, shape =", df.shape)
    df = df.loc[:,CHANNELS_NAME]
    if len(df) == 0: # If all are calibration, do nothing
        return
    
    
    # 3. Applied average montage, bandpass filter                                                     
    df         = average_montage(df)                                                                    # Applied average montage
    filter_df  = mne.filter.filter_data(df.to_numpy().T, sfreq = ORIGINAL_FREQUENCY, l_freq = 0.5, h_freq = 70)  # Applied bandpass filter
    
    # 4. Create a raw mne instance to resample the data to NEW_FREQUENCY=128
    info = mne.create_info(list(AVE_CHANNELS_NAME), 
                           ch_types=['eeg'] * len(AVE_CHANNELS_NAME), 
                           sfreq=ORIGINAL_FREQUENCY)
    info.set_montage('standard_1020', on_missing = 'ignore')
    raw = mne.io.RawArray(filter_df, info)   
    raw.set_eeg_reference()
    raw = raw.resample(sfreq=NEW_FREQUENCY) # Resample the data to new frequency
    df  = raw.to_data_frame() / 1e6         # Get dataframe (Scaling = 1e6, it convert V to µV, therefore, /1e6 when create RawArray)
    
    # 5. Remove some data If the total recording time in second is not integer (is not divisible by NEW_FREQUENCY)
    if df.shape[0] % NEW_FREQUENCY != 0 : 
        num_row = df.shape[0] // NEW_FREQUENCY * NEW_FREQUENCY
        df = df.iloc[:num_row]
    
    
    # 6. Check is there any null value in eeg signal
    if df.isnull().to_numpy().any():
        print(f"{eeg_file} contain NULL value, the function will return")
        return
    else:
        print(f"{eeg_file} does not contain NULL value, the process will continue")
    
    # 7. Save to a csv file
    df.to_csv(f"{CSV_FILE_DIRC}/eeg{num}.csv",index = False) 
    print(f"The dataframe of {eeg_file} have been saved to {CSV_FILE_DIRC}/eeg{num}.csv\n")
    
    return df

# Sub-function for data_pre_processing
def average_montage(data):
    data            = data.to_numpy()                      # Shape = (total_seconds*NEW_FREQUENCY, 19)
    average         = data.mean(axis=1)[:,None]            # Shape = (total_seconds*NEW_FREQUENCY, 1)
    average_montage = data - average                       # Apply average montage
    df              = pd.DataFrame(average_montage, columns=AVE_CHANNELS_NAME) 
    
    return df


def mu_law(x, mu = 255):
    return np.sign(x)* np.log(1+mu * np.abs(x)) / np.log(1+mu)

def inverse_mu_law(x, mu = 255):
    return np.sign(x) *((1+mu) ** np.abs(x) - 1) / mu


class Dataset_Class(Dataset):
    def __init__(self, x, x1,y): 
        self.x = x
        self.x1= x1
        self.y = y
    def __getitem__(self, index):
        return self.x[index], self.x1[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]

class Dataset_Class1(Dataset):
    def __init__(self, x, x1): 
        self.x = x
        self.x1= x1
    def __getitem__(self, index):
        return self.x[index], self.x1[index]
    def __len__(self):
        return self.x.shape[0]

def split_num_train_valid_test(num):

    num_train = int(np.round(num * 0.7))
    num_valid = int(np.round(num * 0.15))
    num_test  = int(np.round(num * 0.15))

    for i in range(10):
        if num_train + num_valid + num_test > num:
            num_train -= 1
        elif num_train + num_valid + num_test < num:
            num_train += 1
        else:
            break
    else:
        print("Unknown error occur",num_train,num_valid,num_test,num)

    return num_train, num_valid, num_test

def get_dataloader(eeg_file_num, get_dataloader=True, shuffle=True, get_dataDWT = True):
    train_data , valid_data , test_data  = [], [], []
    train_label, valid_label, test_label = [], [], []
    for num in eeg_file_num:
        df              = get_df(f'{CSV_FILE_DIRC}/eeg{num}.csv')       # Get the dataframe from the filename
        spike_locations = get_spike_location(num)                       # Get the spike label from the annotation.json
        df              = mu_law(df / NORMALIZE_CONS_1) / NORMALIZE_CONS_2      # Make the range of data in [-1, 1]
        
        # Get the data with fixed length epoch (in torch tensor float32), and its label 
        # label 0 --> No spike, label 1 --> Spike  
        data, label     = get_data_n_label(df, spike_locations) 
        data            = data.permute(0, 2, 1)
        
        # Split the data without/with spike into 70% training, 15% validation, 15% testing
        for label_no in [0, 1]:
            mask_label      = label == label_no
            num_label       = mask_label.sum()
            if num_label > 0:
                num_train, num_valid, num_test = split_num_train_valid_test(num_label)
                data_temp = data[mask_label]
                # print(data_temp, data_temp.shape)
                # print(mask_label, label_no)
                train_data.append(data_temp[:num_train])
                valid_data.append(data_temp[num_train : num_train+num_valid])
                test_data.append(data_temp[num_train+num_valid : num_train+num_valid+num_test])
                train_label.append(torch.full((num_train,), label_no, dtype=torch.int8))
                valid_label.append(torch.full((num_valid,), label_no, dtype=torch.int8))
                test_label.append (torch.full((num_test,),  label_no, dtype=torch.int8))


        print(f"EEG{num} has {data.shape[0]} windows of data \n\n")

    # Concatenate all data
    train_data = torch.cat(train_data, dim=0).to(torch.float32) 
    valid_data = torch.cat(valid_data, dim=0).to(torch.float32) 
    test_data  = torch.cat(test_data, dim=0).to(torch.float32) 
    datasets    = [train_data, valid_data, test_data]
    
    # Concatenate all label
    train_label    = torch.cat(train_label, dim=0).to(torch.float32) 
    valid_label    = torch.cat(valid_label, dim=0).to(torch.float32) 
    test_label     = torch.cat(test_label, dim=0).to(torch.float32) 
    datasets_label = [train_label, valid_label, test_label]
    
    # Get the number of data for each of the dataset
    num_data      = [train_label.shape[0], valid_label.shape[0], test_label.shape[0]]
    
    # Get the data DWT
    datasets_DWT  = []
    for dataset in datasets:

        data_DWT = pywt.wavedec(dataset, 'db1')                         # Apply DWT
        data_DWT = np.concatenate(data_DWT, axis=2, dtype=np.float32)# Concatenate all DWT data
        data_DWT = torch.from_numpy(data_DWT)
        datasets_DWT.append(data_DWT)
        
    # Get the dataloader
    dataloaders = []
    for dataset, dataset_label, dataset_DWT in zip(datasets, datasets_label, datasets_DWT):
        if get_dataDWT:
            dataloader = DataLoader(dataset = Dataset_Class(dataset, dataset_DWT, dataset_label), 
                                    batch_size = BATCH_SIZE, shuffle = shuffle, num_workers=1)
        else:
            dataloader = DataLoader(dataset = Dataset_Class1(dataset, dataset_label), 
                                    batch_size = BATCH_SIZE, shuffle = shuffle, num_workers=1)
        dataloaders.append(dataloader)
        

    print("> > > Train    data  has shape:",datasets[0].shape, "when duration =",DURATION, "seconds")
    print("> > > Data after DWT has shape:",datasets_DWT[0].shape)
    print("> > > Label              shape:",datasets_label[0].shape)
    
    if get_dataloader:
        return dataloaders, num_data
    else:
        if get_dataDWT:
            return datasets, datasets_label, datasets_DWT
        else:
            return datasets, datasets_DWT


def get_spike_location(num):
    if os.path.exists(f"{CSV_FILE_DIRC}/annotation.json"):
        with open(f"{CSV_FILE_DIRC}/annotation.json", "r") as f:
            dict_label = json.load(f)
        if f"eeg{num}" in dict_label.keys():
            spike_list = dict_label[f"eeg{num}"]
            return spike_list        
    return []


# Subfunction for 'get_dataloader'
def get_df(filename, nrows=None):
    if nrows == None:
        df = pd.read_csv(filename, dtype = COL_DTYPE)
    else:
        df = pd.read_csv(filename, dtype = COL_DTYPE, nrows=nrows)

    df    = df.iloc[: , 1:]                        # drop time
    print(f"The data from {filename} is loaded ")
    
    return df


# Subfunction for 'get_dataloader'
def get_data_n_label(data, spike_locations, duration = DURATION):
    data = data.to_numpy()
    if spike_locations != []:  # If this file contain spike, only get the spike
        remove_list = []
        spike_list  = []
        for sl in spike_locations:
            int_sl   = int(sl)
            decimals = sl - int(sl)

            # Increase the spike size with factor of 3
            spike_list.append(np.arange((int_sl-duration//2)*(NEW_FREQUENCY)    , (int_sl+duration//2)*NEW_FREQUENCY))
            spike_list.append(np.arange((int_sl-2-duration//2)*(NEW_FREQUENCY)  , (int_sl-2+duration//2)*NEW_FREQUENCY))
            spike_list.append(np.arange((int_sl+2-duration//2)*(NEW_FREQUENCY)  , (int_sl+2+duration//2)*NEW_FREQUENCY))
            
        # Convert to numpy array    
        spike_list  = np.concatenate(spike_list , axis=0).astype(np.int64)
        
        # Get the data that contain spike
        data_spike     = data[spike_list].copy()
        
        # Concatenate spike and non-spike data
        data = data_spike
        
        # Split the data into windows
        data_windows = []
        for i in range(len(data) // duration // NEW_FREQUENCY):
            data_windows.append(data[None, i*duration*NEW_FREQUENCY:(i+1)*duration*NEW_FREQUENCY,:])
        data_windows = np.concatenate(data_windows, axis=0)
         
        # Get the labels
        labels = np.zeros((len(data_windows), ), dtype=np.int8)
        labels[:len(data_spike)//NEW_FREQUENCY//DURATION]= 1
        
        print("There is spike in this eeg file")
        print("Data before split :", data.shape)
        print("Data with   spike:", data_spike.shape)
        print("Data after  split into window:", data_windows.shape)
        print("Labels:", labels.shape)
        print("Num spike:", labels.sum())
        
    else:  # If there is no spike
        data_windows = []
        for i in range(0, len(data) // duration // NEW_FREQUENCY, 2): # Skip some of the data
            data_windows.append(data[None, i*duration*NEW_FREQUENCY:(i+1)*duration*NEW_FREQUENCY,:])
        data_windows = np.concatenate(data_windows, axis=0)
        labels       = np.zeros((len(data_windows), ), dtype=np.int8)
        
        print("There is no spike in this eeg file")
        print(data_windows.shape)
        
        
    return torch.from_numpy(data_windows).type(torch.float32), torch.from_numpy(labels).type(torch.int8)





@torch.no_grad()
def p_sample_loop(model, shape, label, n_steps = 50, initial_x = None, x_DWT = None, clamp_denoised = True, device=torch.device("cpu")):
    if initial_x == None:
        cur_x = torch.randn(shape).to(device)
    else:
        cur_x = initial_x.to(device)
    x_seq = [cur_x]
    
    for t in reversed(range(n_steps - 1)):
        batch_size  = cur_x.shape[0]
        noise_level = torch.FloatTensor([alphas_prod[t].sqrt()]).repeat(batch_size, 1).to(device)   # Current noise level
        eps_recon   = model(x_DWT, cur_x, noise_level, label)
        mean        = 1/alphas[t].sqrt() * (cur_x - (1-alphas[t]) / (1-alphas_prod[t]).sqrt() * eps_recon)
        std         = (0.5 * posterior_log_variance_clipped[t]).exp()
        eps         = torch.randn_like(cur_x)
        cur_x       = mean + std * eps
        if clamp_denoised:
            cur_x.clamp_(-1.0, 1.0)

    x_seq.append(cur_x)  # only save final result to save the memory used
    return x_seq


def compute_loss(model, x_0, x_0_dwt, label):
    batch_size = x_0.shape[0]   
    eps        = torch.randn_like(x_0)                          # Sample a random error
    
    # Random draw sample from uniform distribution with low = alphas_prod_p_sqrt[t-1], high = alphas_prod_p_sqrt[t] with size = (batch_size,1)
    continuous_sqrt_alpha_cumprod = sample_continuous_noise_level(batch_size).to(device)

    x_noisy = q_sample(x_0, continuous_sqrt_alpha_cumprod, eps) # Diffuse the signal to x_t, given x_0, alpha & random error
    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(batch_size, 1)

    eps_recon_label  = model(x_0_dwt, x_noisy, continuous_sqrt_alpha_cumprod, label)     # Reconstruct the added noise condition on label
    eps_recon_unlabel= model(x_0_dwt, x_noisy, continuous_sqrt_alpha_cumprod, torch.full(label.shape,2, dtype=torch.int64).to(device)) # Error without label
    w = 0.1
    eps_recon = (w + 1)* eps_recon_label - w*eps_recon_unlabel
    
    return torch.nn.L1Loss()(eps_recon, eps)                     # The Lsimple loss 

# Sub-function for 'compute_loss' to return closed form signal diffusion
def q_sample(x_0, continuous_sqrt_alpha_cumprod, eps):
    return continuous_sqrt_alpha_cumprod * x_0 + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * eps

# Sub-function for 'compute_loss' to samples continuous noise level
# (This is what makes WaveGrad different from other Denoising Diffusion Probabilistic Models)
def sample_continuous_noise_level(batch_size):
    t = np.random.choice(range(1, n_steps), size=batch_size)
    continuous_sqrt_alpha_cumprod = torch.FloatTensor(np.random.uniform(alphas_prod_p_sqrt[t-1], alphas_prod_p_sqrt[t], size=batch_size))
    return continuous_sqrt_alpha_cumprod.unsqueeze(-1).unsqueeze(-1)  # Add 1 more dimension (BATCH_SIZE, 1)



def train(model, ema, dataloader, device, num_data, optimizer):
    sum_loss = 0

    for epoch_data, epoch_data_DWT, label in dataloader:
        epoch_data_DWT = epoch_data_DWT.to(device)
        epoch_data     = epoch_data.to(device)
        label          = label.to(device).to(torch.int64)
        
        loss = compute_loss(model, epoch_data, epoch_data_DWT, label)   # Compute the loss.
        sum_loss += loss.detach().cpu().item()                          # Accumulate the loss
        
        optimizer.zero_grad()                                    # Before the backward pass, zero all of the network gradients
        loss.backward()                                          # Backward pass: compute gradient of the loss with respect to parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)   # Perform gradient clipping
        optimizer.step()                                         # Calling the step function to update the parameters
        ema.update(model)                                        # Update the exponential moving average
         
    return sum_loss/num_data

@torch.no_grad()
def evaluate(model, dataloader, device, num_data):
    sum_loss = 0
    
    for epoch_data, epoch_data_DWT, label in dataloader:
        epoch_data_DWT = epoch_data_DWT.to(device)
        epoch_data = epoch_data.to(device)
        label      = label.to(device).to(torch.int64)
        
        loss = compute_loss(model, epoch_data, epoch_data_DWT, label)# Compute the loss.
        sum_loss += loss.detach().cpu().item()                       # Accumulate the loss                                 
        
    return sum_loss/num_data

def print_log(string_to_print, MODEL_FILE_DIRC):
    with open(f'{MODEL_FILE_DIRC}/log.txt', "a") as f:
        print(string_to_print, file=f)
    print(string_to_print)

def train_classifier(model, dataloader, device, num_data, optimizer, LOSS_POS_WEIGHT=torch.tensor([3.031])):
    sum_loss    = 0
    list_pred_label = []
    list_corr_label = []
    loss_func       = nn.BCEWithLogitsLoss(pos_weight=LOSS_POS_WEIGHT.to(device))
    
    for epoch_data, label in dataloader:
        epoch_data = epoch_data.to(device)
        label      = label[:,None].to(device).float()
        
        output    = model(epoch_data)
        loss      = loss_func(output, label)   # Compute the loss.
        sum_loss += loss.detach().cpu().item() # Accumulate the loss
        list_pred_label.extend(torch.sigmoid(output.cpu().detach()).numpy().round().flatten().tolist())
        list_corr_label.extend(label.cpu().flatten().tolist())
        
        optimizer.zero_grad()                                    # Before the backward pass, zero all of the network gradients
        loss.backward()                                          # Backward pass: compute gradient of the loss with respect to parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)   # Perform gradient clipping
        optimizer.step()                                         # Calling the step function to update the parameters

    metrics = {"precision":precision_score(list_corr_label, list_pred_label,zero_division=0), 
               "accuracy":accuracy_score(list_corr_label, list_pred_label), 
               "f1_score":f1_score(list_corr_label, list_pred_label,zero_division=0),
               "recall":recall_score(list_corr_label, list_pred_label,zero_division=0)}
    return sum_loss/num_data, metrics

@torch.no_grad()
def evaluate_classifier(model, dataloader, device, num_data, LOSS_POS_WEIGHT=torch.tensor([3.031])):
    sum_loss = 0
    list_pred_label = []
    list_corr_label = []
    loss_func       = nn.BCEWithLogitsLoss(pos_weight=LOSS_POS_WEIGHT.to(device))
    
    for epoch_data, label in dataloader:
        epoch_data = epoch_data.to(device)
        label      = label[:,None].to(device).float()
        
        output     = model(epoch_data)
        loss       = loss_func(output, label)        # Compute the loss.
        
        sum_loss += loss.detach().cpu().item()       # Accumulate the loss  
        list_pred_label.extend(torch.sigmoid(output.cpu().detach()).numpy().round().flatten().tolist())
        list_corr_label.extend(label.cpu().flatten().tolist())

    metrics = {"precision":precision_score(list_corr_label, list_pred_label,zero_division=0), 
               "accuracy":accuracy_score(list_corr_label, list_pred_label), 
               "f1_score":f1_score(list_corr_label, list_pred_label,zero_division=0),
               "recall":recall_score(list_corr_label, list_pred_label,zero_division=0)}
    return sum_loss/num_data, metrics   # Return the loss and the accuracy


def train_diffusion_classifier(model, dataloader, device, num_data, optimizer, LOSS_POS_WEIGHT=torch.tensor([3.031])):
    sum_loss    = 0
    list_pred_label = []
    list_corr_label = []
    loss_func       = nn.BCEWithLogitsLoss(pos_weight=LOSS_POS_WEIGHT.to(device))
    
    for epoch_data, epoch_data_DWT, label in dataloader:
        epoch_data_DWT = epoch_data_DWT.to(device)
        epoch_data     = epoch_data.to(device)
        label      = label[:,None].to(device).float()
        
        output    = model(epoch_data_DWT, epoch_data)
        loss      = loss_func(output, label)   # Compute the loss.
        sum_loss += loss.detach().cpu().item() # Accumulate the loss
        list_pred_label.extend(torch.sigmoid(output.cpu().detach()).numpy().round().flatten().tolist())
        list_corr_label.extend(label.cpu().flatten().tolist())
        
        optimizer.zero_grad()                                    # Before the backward pass, zero all of the network gradients
        loss.backward()                                          # Backward pass: compute gradient of the loss with respect to parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)   # Perform gradient clipping
        optimizer.step()                                         # Calling the step function to update the parameters
        
    list_pred_label = np.array(list_pred_label)
    list_corr_label = np.array(list_corr_label)
    if np.isnan(list_pred_label).sum(): # If it become nan, it means that it overflow, when computing sigmoid, so replace nan by 1
        list_pred_label = np.nan_to_num(list_pred_label, nan=1)
    
    metrics = {"precision":precision_score(list_corr_label, list_pred_label,zero_division=0), 
               "accuracy":accuracy_score(list_corr_label, list_pred_label), 
               "f1_score":f1_score(list_corr_label, list_pred_label,zero_division=0),
               "recall":recall_score(list_corr_label, list_pred_label,zero_division=0)}
    return sum_loss/num_data, metrics

@torch.no_grad()
def evaluate_diffusion_classifier(model, dataloader, device, num_data, LOSS_POS_WEIGHT=torch.tensor([3.031])):
    sum_loss = 0
    list_pred_label = []
    list_corr_label = []
    loss_func       = nn.BCEWithLogitsLoss(pos_weight=LOSS_POS_WEIGHT.to(device))
    
    for epoch_data, epoch_data_DWT, label in dataloader:
        epoch_data_DWT = epoch_data_DWT.to(device)
        epoch_data = epoch_data.to(device)
        label      = label[:,None].to(device).float()
        
        output = model(epoch_data_DWT, epoch_data)
        loss   = loss_func(output, label)            # Compute the loss.
        
        sum_loss += loss.detach().cpu().item()       # Accumulate the loss  
        list_pred_label.extend(torch.sigmoid(output.cpu().detach()).numpy().round().flatten().tolist())
        list_corr_label.extend(label.cpu().flatten().tolist())

    list_pred_label = np.array(list_pred_label)
    list_corr_label = np.array(list_corr_label)
    if np.isnan(list_pred_label).sum(): # If it become nan, it means that it overflow, when computing sigmoid, so replace nan by 1
        list_pred_label = np.nan_to_num(list_pred_label, nan=1)
    metrics = {"precision":precision_score(list_corr_label, list_pred_label,zero_division=0), 
               "accuracy":accuracy_score(list_corr_label, list_pred_label), 
               "f1_score":f1_score(list_corr_label, list_pred_label,zero_division=0),
               "recall":recall_score(list_corr_label, list_pred_label,zero_division=0)}
    return sum_loss/num_data, metrics   # Return the loss and the accuracy


def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list
