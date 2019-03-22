
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

class MachineLearningModel(object):
    '''
    this is to help separate models by attributes, i.e. IDs to keep track of many models being trained in one batch
    '''
    def __init__(self, model, model_family, model_type, framework, nn=False, id=None):
        self.model = model
        self.framework = framework
        self.model_family = model_family
        self.model_type = model_type
        self.nn = nn
        self.train_sizes = None
        self.train_scores = None
        self.val_scores = None
        self.cm = None
        self.train_cluster_assign = None
        self.train_cluster_proba = None
        self.train_homogeneity = None
        self.test_cluster_assign = None
        self.test_cluster_proba = None
        self.test_homogeneity = None
        if id:
            self.id = id #set as id if provided
        else:
            self.id = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) #otherwise set id to time right now
    def set_training_time(self, training_time):
        self.training_time = round(training_time,4) #training time rounded to 4 decimals
    def set_evaluation_time(self, evaluation_time):
        self.evaluation_time = round(evaluation_time,4) #training time rounded to 4 decimals
    def get_train_sizes(self):
        return self.train_sizes
    def get_train_scores(self):
        if self.model_family != 'NeuralNetwork':
            return np.mean(self.train_scores, axis=1)
        else:
            return self.train_scores
    def get_validation_scores(self):
        if self.model_family != 'NeuralNetwork':
            return np.mean(self.val_scores, axis=1)
        else:
            return self.val_scores
    def get_cm(self):
        return self.cm
    def get_normalized_cm(self):
        return self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
    def __str__(self):
        return 'MODEL DETAILS: ' + self.model_type + ' model from ' + self.framework + ' with ID: ' + str(self.id)


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

def pickle_save_model(algo, model_folder='models'):
    '''
    save the model with datetime if with_datetime=True, which will save model_20190202122930
    '''
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    filename = model_folder+'/'+str(algo.model_type)+'.model'
    print(f'saving as file: {filename}')
    pickle.dump(algo, open(filename, 'wb'))
    return None

def pickle_load_model(model_path):
    '''
    if no model_full_path is provided, then assume that we are looking for the most recent model
    model type can also be specified to retrieve the most recent model of a current type
    '''
    try:
        model = pickle.load(open(model_path, 'rb'))
        print('model successfully loaded from: {}'.format(model_path))
        return model
    except:
        print('did not successfully load model from: {}'.format(model_path))
        FileNotFoundError('model file not found')