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

# importing random forest classifier from ensemble module
from sklearn.ensemble import RandomForestClassifier
# metrics are used to find accuracy or error
from sklearn import metrics  


# 1.Obtain connectivity feature file names 
# Filter only files
folder_path = '../All_Features/'
file_names = [f for f in os.listdir(folder_path)]
connectivity_feature_file_names = [f for f in file_names if 'connectivity' in f]
connectivity_feature_file_names

#2. Load dataset and obtain y labels
file_path = '../../test_data.npz'
# Load the .npz file
loaded_data = np.load(file_path)
#Select y labels 
y_data = loaded_data['y']


#3.Do train-validation split based on saved train / test indices_______________________________________________________________________ 
train_test_indices = joblib.load('train_test_indices.pkl')

#This is the training set that will be used for local parameter selection  _______________________
#Parameter being selected is the connectivity metric to be used
train_idx_full = train_test_indices['train_idx'] # These is an array, where each element is an index number referring to a data sample

#Now split the full training set into a training and validation set __________________________
n_samples = len(train_idx_full)
val_size = 0.25
rng = np.random.default_rng(seed=10)
indices = np.arange(n_samples)
rng.shuffle(indices)

n_val = int(n_samples * val_size)
# This part of the code is slightly different from train-test function, as the indices for train and val need to be selected from train_idx_full
val_idx = train_idx_full[indices[:n_val]]
train_idx = train_idx_full[indices[n_val:]]


# Train and validation labels for connectivity feat. selection
y_train = y_data[train_idx]
y_val = y_data[val_idx]


#4. Use a Random Forest with default parameters to do local parameter selction of best connectivity metric per KnowEEG Paper ________

results_dict = {} # create dictionary to save the results to select the best metric
results_dict['connectivity_data'] = []
results_dict['accuracy'] = []

for file_name in connectivity_feature_file_names:
    
    X = joblib.load('../All_Features/' + file_name)
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    clf = RandomForestClassifier(random_state = 5 , n_jobs = -1)
    clf.fit( X_train , y_train )
    y_pred = clf.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    
    results_dict['connectivity_data'].append(file_name)
    results_dict['accuracy'].append(acc)

results_df = pd.DataFrame(results_dict)

# Select the best performing connectivity metric by accuracy __________
results_df = results_df.sort_values(by = 'accuracy', ascending = False)
selected_metric = results_df.iloc[0,0]
selected_metric

# Save the selected connectivity metric to a file called 'selected_connectivity_metric.txt' _________________
# write
with open("selected_connectivity_metric.txt", "w") as f:
    f.write(selected_metric)