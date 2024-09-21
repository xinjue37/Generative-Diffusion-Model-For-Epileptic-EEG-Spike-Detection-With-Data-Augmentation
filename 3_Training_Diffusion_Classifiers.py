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
from torchsummary import summary
import torch.optim.lr_scheduler as lr_scheduler


LOSS_POS_WEIGHT       = torch.tensor([1])
LEARNING_RATE         = 0.0001
MODEL_FILE_DIRC_WaveGrad = MODEL_FILE_DIRC + "/WaveGrad"
MODEL_FILE_DIRC_DC       = MODEL_FILE_DIRC + "/DC"

torch.manual_seed(3407)

dataloaders, num_data = get_dataloader(list(range(1,98)), shuffle=True)

num_train_data, num_valid_data, num_test_data = num_data
train_data, valid_data, test_data             = dataloaders

def load_classification_model_dict(model, MODEL_FILE_DIRC, model_name):
    list_model = os.listdir(MODEL_FILE_DIRC) 
    if len(list_model) > 0:    # Load the latest trained model
        if os.path.exists(f"{MODEL_FILE_DIRC}/{model_name}_best.pt"):
            state_dict_loaded        = torch.load(f"{MODEL_FILE_DIRC}/{model_name}_best.pt")
            prev_best_valid_f1_score = state_dict_loaded["valid_f1_score"]
            prev_best_valid_loss     = state_dict_loaded["valid_loss"]
        list_model.remove(f"{model_name}_best.pt")
        num_list   = [int(model_dir[model_dir.rindex("_") +1: model_dir.rindex(".")]) for model_dir in list_model if model_dir.endswith(".pt")]
        num_max    = np.max(num_list)
        
        state_dict_loaded = torch.load(f"{MODEL_FILE_DIRC}/{model_name}_{num_max}.pt")
        model.load_state_dict(state_dict_loaded["model"])
        EPOCH_START = state_dict_loaded["epoch"] + 1
        
        print(f"The model has been loaded from the file '{model_name}_{num_max}.pt'")

        if os.path.exists(f"{MODEL_FILE_DIRC}/Loss.csv"):
            df = pd.read_csv(f"{MODEL_FILE_DIRC}/Loss.csv")
            df = df.iloc[:EPOCH_START-1, :]
            print(f"The dataframe that record the loss have been loaded from {MODEL_FILE_DIRC}/Loss.csv")

    else:
        EPOCH_START            = 1
        prev_best_valid_f1_score = -1
        prev_best_valid_loss   = 10000
        df                     = pd.DataFrame(columns = ["Train Loss", "Valid Loss"] + \
                                                         flatten_concatenation([[f"Train {metric}", f"Valid {metric}"] for metric in ["precision", "accuracy", "f1_score", "recall"]]) )
    return model, df, EPOCH_START, prev_best_valid_f1_score, prev_best_valid_loss

def start_classification_model_training(EPOCH_START, NUM_EPOCHS_CLASSIFIER, 
                                        model, MODEL_FILE_DIRC, model_name,  
                                        df, prev_best_valid_f1,prev_best_valid_loss,
                                        train_data, num_train_data, 
                                        valid_data, num_valid_data, 
                                        scheduler, optimizer, device):
    count = 0
    for epoch in range(EPOCH_START, NUM_EPOCHS_CLASSIFIER):
        
            ## 1. Training
            model.train()
            train_loss, train_metric = train_diffusion_classifier(model, train_data, device, num_train_data, optimizer, LOSS_POS_WEIGHT=LOSS_POS_WEIGHT) 
            
            ## 2. Evaluating
            model.eval()
            valid_loss, valid_metric = evaluate_diffusion_classifier(model, valid_data, device, num_valid_data, LOSS_POS_WEIGHT=LOSS_POS_WEIGHT) 
            
            ## 3. Show the result
            list_data       = [train_loss, valid_loss]
            for key in ["precision", "accuracy", "f1_score", "recall"]:
                list_data.append(train_metric[key])
                list_data.append(valid_metric[key])
            df.loc[len(df)] = list_data
            
            print_log(f"> > > Epoch     : {epoch}", MODEL_FILE_DIRC)
            if epoch > 5 and (np.isnan(train_loss) or np.isnan(valid_loss)):
                print("Training break as either train loss or valid loss contain nan")
                break
            print_log(f"Train {'loss':<10}: {train_loss}", MODEL_FILE_DIRC)
            print_log(f"Valid {'loss':<10}: {valid_loss}", MODEL_FILE_DIRC)
            for key in ["precision", "accuracy", "f1_score", "recall"]:
                print_log(f"Train {key:<10}: {train_metric[key]}", MODEL_FILE_DIRC)
                print_log(f"Valid {key:<10}: {valid_metric[key]}", MODEL_FILE_DIRC)

            
            ## 3.1 Plot the loss function
            fig,ax = plt.subplots(3,2, figsize=(10,10))
            x_data = range(len(df["Train Loss"]))
            for i, key in enumerate(["Loss","precision", "accuracy", "f1_score", "recall"]):    
                ax[i%3][i//3].plot(x_data, df[f"Train {key}"], label=f"Train {key}")
                ax[i%3][i//3].plot(x_data, df[f"Valid {key}"], label=f"Valid {key}")
                ax[i%3][i//3].legend()
            plt.savefig(f'{MODEL_FILE_DIRC}/Loss.png', transparent=False, facecolor='white')
            plt.close('all')

            ## 3.3. Save model and Stoping criteria
            if prev_best_valid_f1 <= valid_metric["f1_score"]:  # If previous best validation f1-score <= current f1-score 
                state_dict = {
                    "model": model.state_dict(), 
                    "epoch":epoch,
                    "valid_f1_score": valid_metric["f1_score"],
                    "valid_loss": valid_loss
                }
                torch.save(state_dict, f"{MODEL_FILE_DIRC}/{model_name}_best.pt")
                prev_best_valid_f1 = valid_metric["f1_score"]  # Previous validation loss = Current validation loss
                count = 0
            else:
                count += 1
            
            if epoch % 5 == 0:
                state_dict = {
                    "model": model.state_dict(), 
                    "epoch":epoch,
                    "valid_f1_score": valid_metric["f1_score"],
                    "valid_loss": valid_loss
                }
                torch.save(state_dict, f"{MODEL_FILE_DIRC}/{model_name}_{epoch}.pt")
            
            df.to_csv(f"{MODEL_FILE_DIRC}/Loss.csv", index=False)
            
            if count == MAX_COUNT_F1_SCORE:
                print_log(f"The validation f1 score is not increasing for continuous {MAX_COUNT_F1_SCORE} time, so training stop", MODEL_FILE_DIRC)
                break
            
            scheduler.step()

# Load the pretrained diffusion model
diffusion_model = WaveGradNN(config).to(device)
if os.path.exists(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt"):
    state_dict_loaded    = torch.load(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt")
    diffusion_model.load_state_dict(state_dict_loaded["model"])
else:
    raise("No pretrained diffusion model exists")


# Features gotten when b = [1,2,3,4,5] are
# 512 * 16, 256 * 80, 128 * 320, 128 * 1280, 19  * 1280
# Num of trainable parameters: 1,869,185  1,537,089  1,770,881  2,348,865  1,545,153
for b_ in [1,2,3,4,5]:
    for t_ in [5, 10, 20, 50]:
        print(f"> > Training diffusion classifier model when b = {b_} t = {t_}")
        
        MODEL_FILE_DIRC_DC_bt = MODEL_FILE_DIRC_DC + f"_b{b_}_t{t_}"
        os.makedirs(MODEL_FILE_DIRC_DC_bt, exist_ok=True)
        
        # Create Diffusion Classifier and freeze the layer in diffusion model
        model = DiffusionClassifier(config, diffusion_model, device, t_, b_).to(device)
        for param in model.diffusion_model.parameters():
            param.requires_grad = False

        
        optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)      
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


        # Get the following information:
        # 1. Previous Trained model (if exist)
        # 2. df that store the training/validation loss & metrics
        # 3. epoch where the training start
        # 4. Previous Highest Validation Recall
        # 5. Previous Lowest  Validation Loss
        info_ = load_classification_model_dict(model, MODEL_FILE_DIRC_DC_bt, "DC")
        model = info_[0]
        df    = info_[1] 
        EPOCH_START = info_[2] 
        prev_best_valid_f1_score = info_[3] 
        prev_best_valid_loss     = info_[4]

        # Print out model info
        seperate = "\n" + "-" * 100 + "\n"
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        trainable_params = sum([p.numel() for p in model_parameters])
        print(seperate + "Model infomation" + seperate)
        print(f"Device used       :", device)
        print(f"BATCH SIZE        :", BATCH_SIZE)
        print(f"MAX_COUNT_F1_SCORE:", MAX_COUNT_F1_SCORE)
        print(f"LEARNING RATE     :", LEARNING_RATE)
        print(f"Prev Best recall in validation dataset:", prev_best_valid_f1_score)
        print(f"Prev Best validation loss             :", prev_best_valid_loss)
        print(f"Number of EPOCH for training   :",NUM_EPOCHS_CLASSIFIER, f"(EPOCH start from {EPOCH_START})")
        print(f"Num of epochs of data for train:", num_train_data)
        print(f"Num of epochs of data for valid:", num_valid_data)
        print(f'Model parameters               : {sum(p.numel() for p in model.parameters()):,}' )
        print(f'Trainable Model parameters     : {trainable_params:,}' )
        
        
        # Start training loop
        print(seperate + "Training Log" + seperate)
        start_classification_model_training(EPOCH_START, NUM_EPOCHS_CLASSIFIER, 
                                            model, MODEL_FILE_DIRC_DC_bt, "DC",
                                            df, prev_best_valid_f1_score, prev_best_valid_loss, 
                                            train_data, num_train_data, 
                                            valid_data, num_valid_data, 
                                            scheduler, optimizer, device)
        
        
        # Load the best model and turn to evaluation mode
        model.eval()
        state_dict_loaded = torch.load(f"{MODEL_FILE_DIRC_DC_bt}/DC_best.pt")
        model.load_state_dict(state_dict_loaded["model"])

        test_loss, test_metric = evaluate_diffusion_classifier(model, test_data, device, num_test_data) 

        print_log("Metric on testing dataset:", MODEL_FILE_DIRC_DC_bt)
        for key, value in test_metric.items():
            print_log(f"{key:<10}: {value:.4f}", MODEL_FILE_DIRC_DC_bt)
        print()


df_summary = []
index_list = []
criteria   = "Valid f1_score"

for b_ in [1,2,3,4,5]:
    for t_ in [5, 10, 20, 50]:
    # for t_ in [10]:
        MODEL_FILE_DIRC_DC_bt = MODEL_FILE_DIRC_DC + f"_b{b_}_t{t_}"
        df_loss               = pd.read_csv(f"{MODEL_FILE_DIRC_DC_bt}/Loss.csv")
        df_loss               = df_loss.iloc[:,2:]
        
        index_list.append(f"b{b_}_t{t_}")

        df_summary.append([b_, t_] + df_loss.iloc[np.argmax(df_loss[criteria]),:].tolist())
        # df_summary.append(df_loss.max().tolist())
        
df_col = df_loss.columns.tolist()
df_col = ["b", "t"] + df_col
df_summary = pd.DataFrame(df_summary, columns=df_col, index=index_list)

# Load the pretrained diffusion model
diffusion_model = WaveGradNN(config).to(device)
if os.path.exists(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt"):
    state_dict_loaded    = torch.load(f"{MODEL_FILE_DIRC_WaveGrad}/Advanced_Diffusion_best.pt")
    diffusion_model.load_state_dict(state_dict_loaded["model"])
else:
    raise("No pretrained diffusion model exists")


list_test_metric = []
for b_ in [1,2,3,4,5]:
    for t_ in [5, 10, 20, 50]:
        print(f"Evaluating b={b_}, t={t_}")
        
        MODEL_FILE_DIRC_DC_bt = MODEL_FILE_DIRC_DC + f"_b{b_}_t{t_}"
        
        # Create Diffusion Classifier and freeze the layer in diffusion model
        model = DiffusionClassifier(config, diffusion_model, device, t_, b_).to(device)
        for param in model.diffusion_model.parameters():
            param.requires_grad = False
            
        # Load the best model and turn to evaluation mode
        model.eval()
        state_dict_loaded = torch.load(f"{MODEL_FILE_DIRC_DC_bt}/DC_best.pt")
        model.load_state_dict(state_dict_loaded["model"])

        # Evaluate the best model on the test dataset
        test_loss, test_metric = evaluate_diffusion_classifier(model, test_data, device, num_test_data) 
        
        list_test_metric.append(list(test_metric.values()))

test_metrics = np.array(list_test_metric)

for col, metric in enumerate(test_metric.keys()):
    idx = 4 + col*3
    df_summary.insert(idx, f"Test {metric}", test_metrics[:,col])
    
    print(f"Iteration {col}, Inserting Test {metric:<12} in idx {idx}")


df_summary.to_csv("Diffusion_Classifier.csv", index=False)
df_summary