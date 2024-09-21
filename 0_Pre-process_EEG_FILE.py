import mne
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
from utility import *
import json 
from ummc_db_util import Compumedics


pd.set_option('display.max_rows', 500)

utc              = pytz.UTC
SECONDS_TO_TRASH = 10         # Choosing difference SECONDS_TO_TRASH will have the effect on value inside eeg due to the function (mne.resample) 
dict_spike       = {}
dict_duration    = {}

for filename in os.listdir(EEG_FILE_DIRC):
    if filename.endswith(".eeg"):
            edf_filename = f"{EEG_FILE_DIRC}/" + filename

            dict_spike[edf_filename] = []
            print("For the file:",edf_filename)
            
            # Load the edf file and get the annotation of data
            raw = Compumedics(edf_filename).export_to_mne_raw()  # Return a raw object

            annotation_df     = raw.annotations.to_data_frame()                   # Annotation of the data in dataframe format
            
            # ---------- FIND DURATION TO SKIP ---------- 
            clean_data_before = 0                                                 # Initialize start time as 0

            # Find the calibration duration (if there is any)
            for i, description in enumerate(annotation_df["description"]):  # For each of the descriptions
                if description.lower().__contains__("calibration"):         # If it contain the word 'calibration'
                    print("Description contain 'calibration'",annotation_df.iloc[i,:].to_numpy(), "\n")               

                    # Obtain the start time of the annotation and its duration
                    annotated_start   = annotation_df.iloc[i,0].timestamp()  # Convert to total seconds
                    duration          = annotation_df.iloc[i,1]
                    temp              = annotated_start+duration
                    
                    if temp < 300 and temp > clean_data_before: # If calibration less than 5m=300s, and it is > than initial time
                        clean_data_before = temp
            
            duration_to_skip = clean_data_before        # Duration to throw from the measurement date
            duration_to_skip = np.ceil(duration_to_skip) + SECONDS_TO_TRASH
            print("Duration to skip from the measurement date: ", duration_to_skip)


            # ---------- FIND SPIKE LOCATION (if there is any) ----------    
            for i, description in enumerate(annotation_df["description"]):
                if description.lower().__contains__("spike"):                 # If it contain the word 'spike'
                    print("Description contain 'spike'", annotation_df.iloc[i,:].to_numpy())

                    # Obtain the start time of the spike occur
                    annotated_start   = annotation_df.iloc[i,0].timestamp()
                    dict_spike[edf_filename].append(annotated_start - duration_to_skip)
            dict_duration[edf_filename] = duration_to_skip

count              = 0
num_spike          = 0
file_contain_spike = []
dict_spike_csv     = {}

for i, (filename, spike) in enumerate(dict_spike.items(), 1):
    if spike != []: 
        count+=1
        num_spike += len(spike)
        file_contain_spike.append(filename)
        
        print(filename, f"\teeg{i}\t", len(spike), spike, "\n")
        dict_spike_csv[f"eeg{i}"] = spike
    else:
        dict_spike_csv[f"eeg{i}"] = []
    
print(f"There is {count} file that contain spike, among them, there is total {num_spike} spikes")

for filename, duration in dict_duration.items():
    print(f"For the file: {filename:<35}, Duration to skip from beginning: {duration} s")


with open(f"{CSV_FILE_DIRC}/annotation.json", "w") as f: 
    json.dump(dict_spike_csv, f)

with open(f"{CSV_FILE_DIRC}/annotation.json", "r") as f:
    dict1 = json.load(f)
    print(dict1)


num = 1

for filename in os.listdir(EEG_FILE_DIRC):
    if filename.endswith(".eeg"):
            # Load the edf file and get the annotation of data
            if not os.path.exists(f"{CSV_FILE_DIRC}/eeg{num}.csv"): # If end with .edf, and the file haven't been process
                eeg_filename = f"{EEG_FILE_DIRC}/" + filename
                process_edf(eeg_filename, num, 
                            SKIP_FIRST=dict_duration[edf_filename], 
                            SKIP_LAST=SECONDS_TO_TRASH)    
            num += 1