import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import joblib
import time
import itertools
import sys 

#TS Fresh Parameter Settings and Functions necessary
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import extract_features
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.feature_selection import significance_tests
from tsfresh import feature_selection


#1.Load all TSFresh Features
file_path = '../All_Features/extracted_tsf_features.pkl'
# Load the .pkl
all_TSresh_features = joblib.load(file_path)
all_TSresh_features = all_TSresh_features.dropna(axis=1) # Drop any columns that contain NaN values


#2. Load dataset and obtain y labels (Necessary for Feature filtering)
file_path = '../../test_data.npz'
# Load the .npz file
loaded_data = np.load(file_path)
#Select y labels 
y_data = loaded_data['y']


#3.Do train test split based on saved train / test indices 
train_test_indices = joblib.load('train_test_indices.pkl')
X_train_tsf = all_TSresh_features.loc[train_test_indices['train_idx']]
y_train =  y_data[train_test_indices['train_idx']]

X_test_tsf = all_TSresh_features.loc[train_test_indices['test_idx']]
y_test = y_data[train_test_indices['train_idx']]

#4. Complete filtering based on X_train data only to avoid information leakage
X_train_features_filtered = select_features(X_train_tsf, y_train)

# Use features from above df to created X_test filtered dataframe for testing
X_test_features_filtered = X_test_tsf[X_train_features_filtered.columns]

#5. Save the filtered TSFresh features
joblib.dump(X_train_features_filtered, 'X_train_tsfresh_features_filtered.pkl')
joblib.dump(X_test_features_filtered, 'X_test_tsfresh_features_filtered.pkl')
