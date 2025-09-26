import pandas as pd
import mne as mne
import os 
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
import constants
from IPython.utils import io
import time
import sys
import yasa
from scipy.signal import welch

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts

from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample
import seaborn as sns

from scipy.signal import welch
import yasa
import constants
import numpy as np


# Load the EEG data _____________________________________________
file_path = '../Toy_Data/test_data.npz'
# Load the .npz file
loaded_data = np.load(file_path)
# Select the X_data that will be used to generate the features
X_data = loaded_data['X']

# Specify parameters for the script _______________________________
metrics = ['plv' , 'ppc', 'pli', 'dpli', 'wpli'] # epoched metrics that will be calculated and saved
# Sampling frequency
sfreq = 128
# Parameters for segmenting the data
segment_length = 64
n_electrodes = 14
n_sequence_length = 256
n_segments = n_sequence_length // segment_length 

# Define power bands used for calculating the connectivity ___________
power_bands = {'delta' : (0.5,4) , 'theta' : (4,8) , 'alpha' : (8,12) , 'sigma' : (12,16) , 'beta' : (16,30) , 'gamma' : (30,40) }
# min and max frequency values for each power band
fmin = [float(val[0]) for val in power_bands.values()]
fmax = [float(val[1]) for val in power_bands.values()]


# Calculate connectivity for each sample and save dataframe with features____________________________________

# Loop through each metric and save the file once generated ____________________________________
for metric in metrics:
    
    t1 = time.time()
    
    # Calculate connectivity metric for each sample across power bands and add to row list ______________
    connectivity_df_row_list = []
    
    for sample in X_data : 
        
        # Reshape the data and transpose to get the desired shape (n_segments, 16, segment_length)
        sample = sample.reshape(n_electrodes, n_segments, segment_length).transpose(1, 0, 2)
        
        # Define channel names (replace with your actual channel names if available)
        ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7','Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14']
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg' ) 
        
        # Create MNE EpochsArray object from your data
        data_epo = mne.EpochsArray(sample, info)
        connectivity = spectral_connectivity_epochs(data_epo , method = metric , sfreq=128,fmin=fmin, fmax=fmax, faverage=True )
        
        #3.Create power band connectivity dataframes _______________________________________________________________________
        power_band_connectivity_dfs = {}
        
        for power_band_index, power_band in enumerate(list(power_bands.keys())):
                channel_names = ch_names
                connectivity_data = connectivity.get_data('dense')[:, : , power_band_index]
                
                channel_data = connectivity_data    
                
                # Create an empty DataFrame
                df = pd.DataFrame(index=channel_names, columns=channel_names)
                
                # Fill the DataFrame with connectivity values
                for i in range(len(channel_names)):
                    for j in range(len(channel_names)):
                        channel_1 = channel_names[i]
                        channel_2 = channel_names[j]
                        connectivity_value = channel_data[i, j]
                        df.loc[channel_1, channel_2] = connectivity_value
                        df.loc[channel_2, channel_1] = connectivity_value
                
                df = df.apply(pd.to_numeric)
                power_band_connectivity_dfs[power_band] = df
        
        # 4.Convert the dataframes into a row for dataframe with data from all samples
        # Go through all of the power_bands and add data as a column for a new dataframe with power_band + channel_1 + channel_2 as feature(s)
        new_df_row = {}
        channels = ch_names #This line is here because previous code used channels and new code uses ch_names, I can remove in future
        
        for power_band in list(power_bands.keys()):
            print(power_band)
            df = power_band_connectivity_dfs[power_band]
            for i, channel in enumerate(channels):
                for channel_2 in channels[i+1:]:
                    val = df.loc[channel, channel_2]
                    new_df_row[power_band + '_' + channel + '_' + channel_2] = [val]
        
        new_df = pd.DataFrame.from_dict(new_df_row, orient = 'columns')
        connectivity_df_row_list.append(new_df)
        
    connectivity_df = pd.concat(connectivity_df_row_list)
    
    #Save the features________________
    file_name = 'connectivity_epoched_' + metric + '.pkl' 
    file_path = '../Toy_Data/Feature_Generation/All_Features/' + file_name
    joblib.dump(connectivity_df, file_path)