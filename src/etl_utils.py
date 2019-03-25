
import pandas as pd 
import datetime
import pickle
import os
import glob
import numpy as np

from sklearn import preprocessing

from sklearn.utils import shuffle, resample

class Dataset(object):
    '''
    this class is used to more easily structure classification datasets between the data (matrix of values) and the target (class)
    instead of doing any indexing when training an algorithm, simply use Dataset.data and Dataset.target
    '''
    def __init__(self, df, target_col_name):
        self.target_col_name = target_col_name
        self.target = df.loc[:, target_col_name].values.astype('int32')
        self.data = df.loc[:, df.columns != target_col_name].values.astype('float32')
        self.pca_data = None
        self.ica_data = None 
        self.srp_data = None 
        self.rffs_data = None
        self.df = df
    def __str__(self):
        return ('first 10 data points\n' + str(self.data.head(10)) + '\nfirst 10 labels\n' + str(self.target.head(10)) + '\n')

def clean_and_scale_dataset(dirty_df_dict, na_action='mean', scaler=None, class_col='class'):
    
    clean_df_list = []

    if scaler:
        scaler.fit(dirty_df_dict['train'].loc[:,dirty_df_dict['train'].columns!=class_col])
    for df_name, dirty_df in dirty_df_dict.items():
        if scaler:
            dirty_df.loc[:,dirty_df.columns!=class_col] = scaler.transform(dirty_df.loc[:,dirty_df.columns!=class_col])

        #how to handle na values in dataset:
        if na_action == 'drop':
            dirty_df = dirty_df.dropna()
        if na_action == 'mean':
            dirty_df = dirty_df.fillna(dirty_df.mean())
        if na_action == 'mode':
            dirty_df = dirty_df.fillna(dirty_df.mode())
        if na_action == 'zeros':
            dirty_df = dirty_df.fillna(0)
        else:
            try:
                dirty_df = dirty_df.fillna(int(na_action))
            except:
                dirty_df = dirty_df.fillna(0) 
                print('filled with zeros as a failover')

        cleaned_df = dirty_df
        clean_df_list.append(cleaned_df)
    
    return clean_df_list

def balance(df, class_col='class', balance_method='downsample'):

    if type(balance_method) == int:
        n_samples = balance_method
    elif type(balance_method) == str:
        if balance_method == 'downsample':
            n_samples = min(df[class_col].value_counts())
        elif balance_method == 'upsample':
            n_samples = max(df[class_col].value_counts())
        else:
            raise ValueError('no viable sampling method provided, please enter (upsample, downsample, or an integer)')

    df_list = []
    for label in np.unique(df[class_col]):
        subset_df = df[df[class_col]==label]
        resampled_subset_df = resample(subset_df, 
                                        replace=(subset_df.shape[0]<n_samples),    # sample with replacement if less than number of samples, otherwise without replacement
                                        n_samples=n_samples)    # to match minority class
        df_list.append(resampled_subset_df)
    balanced_df = pd.concat(df_list)
    
    return balanced_df


def prep_data(df_dict, shuffle_data=True, balance_method='downsample', class_col='class'):
    '''
    always pass training set as first df in list
    '''
    #encode dataset to binary variables
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df_dict['train'][class_col])

    prepped_df_list = []
    
    for df_key, df in df_dict.items():

        #encode training dataset
        df[class_col] = encoder.transform(df[class_col])

        if balance_method:
            if df_key=='train': #only balance training data
                df = balance(df, class_col=class_col, balance_method=balance_method)

        dataset_df = Dataset(df, class_col)
        if shuffle_data:
            dataset_df.data, dataset_df.target = shuffle(dataset_df.data, dataset_df.target)
        
        prepped_df_list.append(dataset_df)

    return prepped_df_list, encoder
