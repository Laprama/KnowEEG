import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas as pd


def calc_corr(X_input, corr_type, num_electrodes = 14):
    '''
    Inputs : - X_raw --> Numpy array with dimmensions  (Num samples, Num electrodes (default 14), Dataset length (256 samples))
             - corr_type --> 's' for spearman , 'p' for pearson
    Output: X_features --> 91 features x N samples (dataframe)
    '''
    # Create the data dictionary ___________
    connectivity_data_dict = {}
    
    for i in range(num_electrodes):
        for j in range(i+1,num_electrodes):
            connectivity_data_dict[corr_type + str(i) + '_' + str(j)] = []

    # Calculate the correlation features _________
    for sample in X_input : 
        nums = []
        for i in range(num_electrodes):
            for j in range(i+1,num_electrodes):
                if corr_type == 's':
                    corr, p_value = spearmanr(sample[i], sample[j])
                    connectivity_data_dict[corr_type + str(i) + '_' + str(j)].append(corr)
                    
                elif corr_type == 'p':
                    corr, p_value = pearsonr(sample[i], sample[j])
                    connectivity_data_dict[corr_type + str(i) + '_' + str(j)].append(corr)

                
                else:
                    raise ValueError(f"Invalid corr_type '{corr_type}'. Expected 's' for Spearman or 'p' for Pearson.")
                    
           
            
    X_connectivity = pd.DataFrame(connectivity_data_dict)

    return X_connectivity



def convert_sktime_df_to_ts_fresh_format(input_df , ts_cols):
    '''
    This function takes in sktime format dataframe and converts to TS fresh format dataframe that can then be used for feature
    extraction. 
    Sktime format dataframe: Each row refers to an entire time series, each column is one variable in the ts. Each cell within each column is a univariate time series.
    index | dimension_0 | ... dimension_n -> index is unique per time series , values in dimension_n are an entire time series
    
    TS Fresh format dataframe (Flat dataframe / Wide DataFrame): id | Time | variable_1 |...variable_n| ....7
    Where each id refers to one time series, Time runs from 1 to length of time series , variable_n is the variable value at that point in time for that id
       
    Input: 
    1. input_df -> dataframe in sktime format
    2. ts_cols -> Specify which columns to use as time series variables 
    Output: Dataframe in TS Fresh format for input data 
    
    Assumption: index in input dataframe uniquely identifies each time series
    
    '''
    #This code converts the sktime format dataframe dfs into TS Fresh format dataframe df
    #Create one dataframe for each time series, therefore ID the same within each dataframe then concatenate them all together to create output dataframe
    
    #list to store each dataframe as they are created
    df_list = []
    

    for ind_val in input_df.index.values:
        frame = {}
        frame['id'] = ind_val

        for col in ts_cols:
            frame[col] = input_df.loc[ind_val][col]

        #This column is generated from the length of one of the time series
        frame['time'] = [i for i in range(len(frame[col]))]

        df_new = pd.DataFrame(frame)
        df_new.reset_index(drop = True)
        df_list.append(df_new)

    df = pd.concat(df_list, axis = 0)
    df = df.reset_index(drop = True)
    
    return df