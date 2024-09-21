import numpy as np
import pandas as pd
import seaborn as sns
import mne
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pywt 
from PIL import Image
from utility import *
from model import *

MODEL_FILE_DIRC_WaveGrad = MODEL_FILE_DIRC + "/WaveGrad"
os.makedirs(MODEL_FILE_DIRC_WaveGrad, exist_ok=True)

torch.manual_seed(3407)

dataloaders, num_data = get_dataloader(list(range(1,98)), shuffle=True)
    
num_train_data, num_valid_data, num_test_data = num_data
train_data, valid_data, test_data             = dataloaders

model     = WaveGradNN(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  
ema       = EMA(0.9) 
ema.register(model)


list_model = os.listdir(MODEL_FILE_DIRC_WaveGrad)
if len(list_model) > 0:    # Load the latest trained model
    if os.path.exists(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt"):
        state_dict_loaded    = torch.load(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt")
        prev_best_valid_loss = state_dict_loaded["valid_loss"]
    list_model.remove("Advanced_Diffusion_best.pt")
    num_list   = [int(model_dir[model_dir.rindex("_") +1: model_dir.rindex(".")]) for model_dir in list_model if model_dir.endswith(".pt")]
    num_max    = np.max(num_list)
    
    state_dict_loaded = torch.load(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_{num_max}.pt")
    model.load_state_dict(state_dict_loaded["model"])
    ema.load_state_dict(state_dict_loaded["model"])
    EPOCH_START = state_dict_loaded["epoch"] + 1
    
    print(f"The model has been loaded from the file 'Advanced_Diffusion_{num_max}.pt'")

    if os.path.exists(f"{MODEL_FILE_DIRC_WaveGrad}/Loss.csv"):
        df = pd.read_csv(f"{MODEL_FILE_DIRC_WaveGrad}/Loss.csv")
        df = df.iloc[:EPOCH_START-1, :]
        print(f"The dataframe that record the loss have been loaded from {{MODEL_FILE_DIRC_WaveGrad}}/Loss.csv")

else:
    EPOCH_START          = 1
    prev_best_valid_loss = 10000 
    df                   = pd.DataFrame(columns = ["Train Loss", "Valid Loss"])

count = 0

seperate = "\n" + "-" * 100 + "\n"
print(seperate + "Model infomation" + seperate)
print(f"Device used    :", device)
print(f"BATCH SIZE     :", BATCH_SIZE)
print(f"MAX_COUNT      :", MAX_COUNT) 
print(f"LEARNING RATE  :", LEARNING_RATE)
print(f"Previous best validation loss  :", prev_best_valid_loss)
print(f"Number of EPOCH for training   :",NUM_EPOCHS, f"(EPOCH start from {EPOCH_START})")
print(f"Num of epochs of data for train:", num_train_data)
print(f"Num of epochs of data for valid:", num_valid_data)
print(f'Model parameters               : {sum(p.numel() for p in model.parameters()):,}' )

def print_log(string_to_print):
    with open(f'{MODEL_FILE_DIRC_WaveGrad}/log.txt', "a") as f:
        print(string_to_print, file=f)
    print(string_to_print)

for epoch in range(EPOCH_START, NUM_EPOCHS):
    
        ## 1. Training
        model.train()
        train_loss = train(model, ema, train_data, device, num_train_data, optimizer) 
        
        ## 2. Evaluating
        model.eval()
        valid_loss = evaluate(model, valid_data, device, num_valid_data) 
            
        ## 3. Show the result
        df.loc[len(df)] = [train_loss, valid_loss]
        print_log(f"Epoch     : {epoch}")
        print_log(f"Train loss: {train_loss}")
        print_log(f"Valid loss: {valid_loss}")
        
        # After 30 iteration, Random Get  1 epoch of data from validation loader, 
        # and visualize the reconstructed output
        if epoch >= 30 and epoch % 10 == 0:
            plt.ioff()
            model = model.to("cpu")
            diffuse_num_step     = 100
            reconstruct_num_step = 100

            iterator = iter(valid_data)
            original_signal, signal_DWT, label = next(iterator)
            rand_num = np.random.randint(0, original_signal.shape[0])
            x_0 = original_signal[rand_num][None,:]
            x_0_DWT = signal_DWT[rand_num][None,:]
            label        = label[rand_num].item()
            eps          = torch.randn_like(x_0)
            initial_x    = q_sample(x_0, alphas_prod_p_sqrt[diffuse_num_step-1], eps)

            x_seq   = p_sample_loop(model, 
                                    x_0.shape, 
                                    label   = torch.ones((1), dtype=torch.int64) if label==1 else torch.zeros((1), dtype=torch.int64),
                                    n_steps = reconstruct_num_step, 
                                    initial_x = initial_x,
                                    x_DWT = x_0_DWT, 
                                    device=torch.device("cpu"))  

            # Reverse the label
            x_seq1   = p_sample_loop(model, 
                                    x_0.shape, 
                                    label   = torch.zeros((1), dtype=torch.int64) if label==1 else torch.ones((1), dtype=torch.int64),
                                    n_steps = reconstruct_num_step, 
                                    initial_x = initial_x,
                                    x_DWT = x_0_DWT, 
                                    device=torch.device("cpu"))
            fig, axs = plt.subplots(len(AVE_CHANNELS_NAME), 1, figsize=(20, 50))

            sample = "Annotated" if label == 1 else "Un-annotated"
            seperate = 1
            for i, col in enumerate(AVE_CHANNELS_NAME):
                axs[i].set_title(col + sample)
                axs[i].plot(x_0[0][i], label=f"Original Signal")
                axs[i].plot(x_seq[0][0, i] + seperate, label=f"Noisy Signal_{diffuse_num_step}")
                axs[i].plot(x_seq[-1][0, i] - seperate, label=f"Reconstructed Signal_{reconstruct_num_step}")
                axs[i].plot(x_seq1[-1][0, i] - seperate*2,label=f"Reconstructed Signal_{reconstruct_num_step}_if_label_opposite")
                axs[i].legend()

            SAVE_PATH = f'{MODEL_FILE_DIRC_WaveGrad}/Epoch_{epoch}_Signal.png'
            plt.savefig(SAVE_PATH, transparent=False, facecolor='white')

            fig, axs = plt.subplots(len(AVE_CHANNELS_NAME), 1, figsize=(20, 50))
            seperate = 1e-4
            for i, col in enumerate(AVE_CHANNELS_NAME):
                axs[i].set_title(col + sample)
                axs[i].plot(inverse_mu_law(x_0[0][i] * NORMALIZE_CONS_2) * NORMALIZE_CONS_1, label=f"Original Signal")
                axs[i].plot(inverse_mu_law(x_seq[-1][0, i] * NORMALIZE_CONS_2) * NORMALIZE_CONS_1 - seperate, label=f"Reconstructed Signal_{reconstruct_num_step}")
                axs[i].plot(inverse_mu_law(x_seq1[-1][0, i] * NORMALIZE_CONS_2) * NORMALIZE_CONS_1 - seperate*2,
                            label=f"Reconstructed Signal_{reconstruct_num_step}_if_label_opposite")
                axs[i].legend()

            SAVE_PATH1 = f'{MODEL_FILE_DIRC_WaveGrad}/Epoch_{epoch}_Original_Signal.png'
            plt.savefig(SAVE_PATH1, transparent=False, facecolor='white')

            model = model.to("cuda")
            plt.close('all')
        
        ## 4.4 Plot the loss function
        plt.plot(range(len(df["Train Loss"])), df["Train Loss"], label="Train Loss")
        plt.plot(range(len(df["Train Loss"])), df["Valid Loss"], label="Valid Loss")
        plt.legend()
        plt.savefig(f'{MODEL_FILE_DIRC_WaveGrad}/Loss.png', transparent=False, facecolor='white')
        plt.close('all')

        ## 4.5. Save model and Stoping criteria
        if prev_best_valid_loss > valid_loss:  # If previous validation loss larger than current validation loss (The model is performed better)
            state_dict = {
                "model": model.state_dict(), 
                "epoch":epoch,
                "valid_loss": valid_loss
            }
            torch.save(state_dict, f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt")
            
            prev_best_valid_loss = valid_loss  # Previous validation loss = Current validation loss
            count = 0
        else:
            count += 1
        
        if epoch % 10 == 0:
            state_dict = {
                "model": model.state_dict(), 
                "epoch":epoch,
                "valid_loss": valid_loss
            }
            torch.save(state_dict, f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_{epoch}.pt")
        
        df.to_csv(f"{MODEL_FILE_DIRC_WaveGrad}/Loss.csv", index=False)
        
        if count == MAX_COUNT:
            print_log(f"The validation loss is continuous decrease for {MAX_COUNT} time, so training stop")
            break