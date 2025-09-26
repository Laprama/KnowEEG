import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import time

#TS Fresh Parameter Settings
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import extract_features

from tsfresh import extract_features, extract_relevant_features, select_features
import format_eeg_data
import joblib

from helper_functions import convert_sktime_df_to_ts_fresh_format

# Script Parameter________
num_electrodes = 14

# Load the data dictionary and specify key to access EEG data 
file_path = '../Toy_Data/test_data.npz'
data_dict = np.load(file_path)
key = 'X'


# Transform the data into the format required by TSFresh __________________________________________
df_dict = {}
for i in range(num_electrodes):
    df_dict[i] = []

# select the correct data
for row in data_dict[key]:
    i = 0
    for electrode_ts in row:
        
        row_series = np.array(electrode_ts)
        row_series = electrode_ts

        df_dict[i].append(row_series)
        i+=1
        
# Turn the dictionary into a dataframe
full_ts = pd.DataFrame(df_dict)
full_ts['id_val'] = full_ts.index

# Call helper function to create the TSFresh format dataframe from full_ts 
df_tsf_format  = format_eeg_data.convert_sktime_df_to_ts_fresh_format(full_ts, full_ts.columns[:-1] )

#Generate TSFresh Features using df_tsf_format dataframe_______________________________________________
settings = EfficientFCParameters()
extracted_ts_fresh_df = extract_features(df_tsf_format, column_id = 'id',  column_sort = 'time',  default_fc_parameters=settings)

#Save the features________________
file_path = '../Toy_Data/Feature_Generation/All_Features/extracted_tsf_features.pkl'
joblib.dump(extracted_ts_fresh_df, file_path)