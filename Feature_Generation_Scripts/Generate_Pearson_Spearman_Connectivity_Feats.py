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

from helper_functions import calc_corr


# Load the EEG data _____________________________________________
file_path = '../Toy_Data/test_data.npz'
# Load the .npz file
loaded_data = np.load(file_path)
# Select the X_data that will be used to generate the features
X_data = loaded_data['X']

df_p = calc_corr( X_data , 'p')
df_s = calc_corr( X_data , 's')

df_correlation = pd.concat([df_p, df_s], axis = 1)

#Save the features________________
file_name = 'connectivity_one_epoch_' + 'correlation' + '.pkl' 
file_path = '../Toy_Data/Feature_Generation/All_Features/' + file_name
joblib.dump(df_correlation, file_path)