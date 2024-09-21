from utility import *
from settings import *
from model import EMA, WaveGradNN, SateLight, CNN
import logging
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.linalg import sqrtm
import numpy as np

MODEL_FILE_DIRC_CNN = MODEL_FILE_DIRC + "/CNN"
torch.manual_seed(3407)

device        = torch.device('cpu')
seperate      = "\n" + "-" * 100 + "\n"
logging.basicConfig(level=20, format =seperate + "     %(levelname)s : PID:%(process)d : %(asctime)s : %(message)s" + seperate )
configuration = {"batch_size":BATCH_SIZE, "max_count":MAX_COUNT, "learning_rate":LEARNING_RATE}

MODEL_FILE_DIRC_WaveGrad = MODEL_FILE_DIRC + "/WaveGrad"

model     = WaveGradNN(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  
ema       = EMA(0.9) 
ema.register(model)

list_model = os.listdir(MODEL_FILE_DIRC_WaveGrad)
if len(list_model) > 0:    # Load the latest trained model
    if os.path.exists(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt"):
        state_dict_loaded    = torch.load(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt")
        model.load_state_dict(state_dict_loaded["model"])   
        ema.load_state_dict(state_dict_loaded["model"])
        
        print(f"The model has been loaded from the file 'Advanced_Diffusion_best.pt'")
    else:
        list_model.remove("Advanced_Diffusion_best.pt")
        num_list   = [int(model_dir[model_dir.rindex("_") +1: model_dir.rindex(".")]) for model_dir in list_model if model_dir.endswith(".pt")]
        num_max    = np.max(num_list)
        
        state_dict_loaded = torch.load(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_{num_max}.pt")
        model.load_state_dict(state_dict_loaded["model"])
        ema.load_state_dict(state_dict_loaded["model"])
        EPOCH_START = state_dict_loaded["epoch"]
        
        print(f"The model has been loaded from the file 'Advanced_Diffusion_{num_max}.pt'")

else:
    raise("No pretrained model exist in the folder 'Model' ")

datasets, datasets_label, datasets_DWT = get_dataloader(list(range(1,98)), get_dataloader=False, shuffle=False)
    
train_dataset = datasets[0]
train_label   = datasets_label[0].type(torch.int64) 
train_DWT     = datasets_DWT[0]


num_step_used_list = [500]
num_split          = 10


synthesized_data = []
for num_step_used in num_step_used_list:
    print(f"> Start for step {num_step_used}")
    
    for i in range(num_split):
        print(f"Iteration {i}")
        start_idx = i * len(train_dataset) // num_split
        end_idx   = (i+1)*  len(train_dataset) // num_split
        x_0       = train_dataset[start_idx:end_idx]
        x_0_DWT   = train_DWT[start_idx:end_idx]
        label     = train_label[start_idx:end_idx].type(torch.int64) 
        eps       = torch.randn_like(x_0)
        
        diffuse_num_step      = num_step_used
        reconstruct_num_step  = num_step_used
        
        x_seq   = p_sample_loop(model, 
                                x_0.shape, 
                                label   = label,
                                n_steps = reconstruct_num_step, 
                                x_DWT   = x_0_DWT, 
                                device=torch.device("cpu"))  
        synthesized_data.append(x_seq[-1])


synthesized_data_test = torch.cat(synthesized_data, axis=0)
print(synthesized_data_test.shape)

# Reshape the signal
bs, num_channel, num_signal = synthesized_data_test.shape
synthesized_data_test       = synthesized_data_test.permute(2,0,1).reshape(bs*num_signal,num_channel)

# Save into csv file
df = pd.DataFrame(synthesized_data_test, columns=AVE_CHANNELS_NAME)
df.to_csv(f"synthesized_data.csv", index=False)