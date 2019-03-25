import pandas as pd 
import datetime
import time

import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import homogeneity_score, roc_auc_score, confusion_matrix

from plot_utils import plot_multi_lines, gen_plot, accumulate_subplots
from etl_utils import *

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA 
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from scipy.stats import kurtosis

import scipy.sparse as sps
from scipy.linalg import pinv

#general params:
DATASETS = ('aps', 'spam') #any of of ('spam', 'aps')

RANDOM_STATE = 27
PLOT_ACTION = None # (None, 'save', 'show') - default to None to avoid issues with matplotlib depending on OS
N_EPOCHS = 10000 # maximum number of epochs for neural network training
N_ITER_ICA = 1000 #max number of iterations for ICA before stopping
N_CLUSTERS = 30 #max number of clusters to try for part 1 of homework
BALANCE_METHOD = 'downsample' # (int, 'downsample' or 'upsample') for training data


def reconstructionError(projections,X):
    '''
    stolen direction from: https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/helpers.py
    '''
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

def generate_clustering_algorithms(id, cluster_list=[2,3,4,5,6,7,8,9,10], cluster_type='kmeans'):
    kmeans_models = {}
    for cluster_size in cluster_list:
        if cluster_type == 'kmeans':
            clusterer = KMeans(n_clusters=cluster_size, random_state=RANDOM_STATE)
        elif cluster_type == 'em':
            clusterer = GaussianMixture(n_components=cluster_size, random_state=RANDOM_STATE)
        kmeans_models[cluster_size] = clusterer
    return kmeans_models

def run_cluster_variations(id, train_dataset, test_dataset, clustering_model_list=['kmeans', 'em'], max_num_cluster=30):
    cluster_list = [x for x in range(2, max_num_cluster+1)]
    
    cluster_models_dict = {}
    
    for clustering_model in clustering_model_list:
        kmeans_models = generate_clustering_algorithms(id, cluster_list, cluster_type=clustering_model)
        train_homogeneity_list = []
        test_homogeneity_list = []
        model_list = []
        num_cluster_list = []
        for num_cluster, algo in kmeans_models.items():
            
            algo_start_time = time.time()
            algo.fit(train_dataset.data)
            algo_elapsed_time = round(time.time() - algo_start_time,2)
            print('{} with {} clusters/components training time: {:.2f} s'.format(clustering_model.upper(), num_cluster, algo_elapsed_time), end='\r', flush=True)
            if clustering_model == 'kmeans':
                algo.train_cluster_assign = algo.labels_
            elif clustering_model == 'em':
                algo.train_cluster_assign = algo.predict(train_dataset.data)

            num_cluster_list.append(num_cluster)
            algo.train_homogeneity = homogeneity_score(train_dataset.target, algo.train_cluster_assign)
            train_homogeneity_list.append(algo.train_homogeneity)

            algo.test_cluster_assign = algo.predict(test_dataset.data)
            algo.test_homogeneity = homogeneity_score(test_dataset.target, algo.test_cluster_assign)
            test_homogeneity_list.append(algo.test_homogeneity)

            model_list.append(algo)
        print('')

        max_index = np.argmax(test_homogeneity_list)
        max_value = test_homogeneity_list[max_index]
        best_cluster = num_cluster_list[max_index]
        print('best homogeneity: {:.2f} achieved using {} with {}'.format(max_value, clustering_model.upper(), best_cluster))

        cluster_models_dict[clustering_model] = {'model_list':model_list, 
                                                'cluster_list':cluster_list, 
                                                'train_homogeneity_list':train_homogeneity_list, 
                                                'test_homogeneity_list':test_homogeneity_list}

    plot_multi_lines(cluster_models_dict, 
                        x_key='cluster_list', 
                        train_y_key='train_homogeneity_list', 
                        test_y_key='test_homogeneity_list', 
                        title_name='Homogeneity by Cluster Size',
                        ylabel_name='Homogeneity',
                        xlabel_name='Number of Clusters',
                        figure_action=PLOT_ACTION, 
                        figure_path='output/'+str(id)+'/part1/figures',
                        file_name='clustering_models')

    return cluster_models_dict


def reverse_sort_by_importance(data, ranking_list, n_features=None):
    rev_arg_sort_indices = np.argsort(ranking_list)[::-1]

    reverse_sorted_data = data[:, rev_arg_sort_indices]
    if n_features is not None:
        return reverse_sorted_data[:,n_features]
    else:
        return reverse_sorted_data
    

def run_feature_selection(id, train_dataset, test_dataset, param_variations=None, models_to_run=['pca', 'ica', 'srp', 'rffs'], n_rp_runs=10):
    '''
    PCA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - vary the number of principle components - see this plot: https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
    ICA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
    Randomized Projections (https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html)
    Other feature selection algorithm (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
    CONSIDER ADDING THIS TO THE DATASET OBJECT, creating a `get_pca(components=4)` or a `get_ica(components=4)` or `get_rp(components=4) or `get_dt_fs(n_components(threshold=-np.inf, max_features=4)
    '''
    plot_locs = [(0,0), (0,1), (0,2), (0,3)]
    fs_details = {}
    if 'pca' in models_to_run:
        pca_model = PCA(random_state=RANDOM_STATE)
        pca_start_time = time.time()
        pca_model.fit(train_dataset.data)
        pca_elapsed_time = (time.time() - pca_start_time)
        print('PCA elapsed time: {:.2f} s'.format(pca_elapsed_time))
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
        pca_variance = pca_model.explained_variance_ratio_
        n_components = np.linspace(1,pca_variance.shape[0],pca_variance.shape[0])
        pca_plot_data = np.column_stack((n_components, pca_variance, cumulative_variance))

        fs_details[plot_locs[0]] = {'type':'line', 
                                    'data_dict':{'ICA':{'x':pca_plot_data[:,0], 'y':pca_plot_data[:,1]}}, 
                                    'title':'PCA',
                                    'ylabel':'Variance',
                                    'xlabel':'Components'}
        
        max_index = np.argmax(pca_variance)
        max_value = pca_variance[max_index]
        print('PCA - highest variance: {:.2f}'.format(max_value))

        
    if 'ica' in models_to_run:
        ica_model = FastICA(max_iter=N_ITER_ICA, tol=0.0001, random_state=RANDOM_STATE)
        # THIS THREW AN ERROR ABOUT NaNs once
        successful = False
        m = 0
        while m < 5 and not successful:
            try:
                ica_start_time = time.time()
                transformed_data = ica_model.fit_transform(train_dataset.data)
                successful=True
                ica_elapsed_time = (time.time() - ica_start_time)
                print('ICA elapsed time: {:.2f} s'.format(ica_elapsed_time))
            except:
                print('ICA got an infinity or NaN value, trying again')
            m += 1

        kurtosis_score_for_all_components = kurtosis(transformed_data, fisher=False)
        rev_arg_sort_indices = np.argsort(kurtosis_score_for_all_components)[::-1]
        rev_sorted_kurtosis_score = kurtosis_score_for_all_components[rev_arg_sort_indices]

        n_components = np.linspace(1,rev_sorted_kurtosis_score.shape[0],rev_sorted_kurtosis_score.shape[0])
        ica_plot_data = np.column_stack((n_components, rev_sorted_kurtosis_score, rev_arg_sort_indices))

        fs_details[plot_locs[1]] = {'type':'line', 
                                    'data_dict':{'ICA':{'x':ica_plot_data[:,0], 'y':ica_plot_data[:,1]}}, 
                                    'title':'ICA',
                                    'ylabel':'Kurtosis',
                                    'xlabel':'Components'}
        
        print('ICA - highest kurtosis: {:.2f}'.format(rev_sorted_kurtosis_score[0]))

    if 'srp' in models_to_run:
        srp_x_data_list = []
        srp_y_data_list = []
        srp_data_dict = {}
        for i in range(n_rp_runs):
            srp_model = SparseRandomProjection(n_components=train_dataset.data.shape[1])

            srp_start_time = time.time()
            transformed_data = srp_model.fit_transform(train_dataset.data)
            srp_elapsed_time = (time.time() - srp_start_time)
            print('SRP elapsed time: {:.2f} s'.format(srp_elapsed_time), end='\r', flush=True)

            kurtosis_score_for_all_components = kurtosis(transformed_data, fisher=False)
            rev_sort_arg_indices = np.argsort(kurtosis_score_for_all_components)[::-1]
            rev_sorted_kurtosis_score = kurtosis_score_for_all_components[rev_sort_arg_indices]

            n_components = np.linspace(1,rev_sorted_kurtosis_score.shape[0],rev_sorted_kurtosis_score.shape[0])
            srp_plot_data = np.column_stack((n_components, rev_sorted_kurtosis_score, rev_sort_arg_indices))

            srp_data_dict['Run {}'.format(i)] = {'x':srp_plot_data[:,0], 'y':srp_plot_data[:,1]}
            
        fs_details[plot_locs[2]] = {'type':'line', 
                                    'data_dict':srp_data_dict, 
                                    'title':'RP',
                                    'ylabel':'Kurtosis',
                                    'xlabel':'Components'}
        print('')
        print('SRP - highest kurtosis: {:.2f}'.format(rev_sorted_kurtosis_score[0]))
                
    if 'rffs' in models_to_run:
        # random forest feature selection
        rffs_model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=RANDOM_STATE)
        rffs_start_time = time.time()
        feature_importances = rffs_model.fit(train_dataset.data,train_dataset.target).feature_importances_ 
        rffs_elapsed_time = (time.time() - rffs_start_time)
        print('RF elapsed time: {:.2f} s'.format(rffs_elapsed_time))

        rev_sort_arg_indices = np.argsort(feature_importances)[::-1]
        rev_sorted_feature_importances = feature_importances[rev_sort_arg_indices]

        n_components = np.linspace(1,rev_sorted_feature_importances.shape[0],rev_sorted_feature_importances.shape[0])
        rfs_plot_data = np.column_stack((n_components, rev_sorted_feature_importances, rev_sort_arg_indices))

        fs_details[plot_locs[3]] = {'type':'line', 
                                    'data_dict':{'RFFS':{'x':rfs_plot_data[:,0], 'y':rfs_plot_data[:,1]}}, 
                                    'title':'RF',
                                    'ylabel':'Feature Importance',
                                    'xlabel':'Components'}
                                    
        print('RFFS - highest feature importance: {:.2f}'.format(rev_sorted_feature_importances[0]))

    accumulate_subplots(subplot_shape=(1,4), 
                        subplot_dict=fs_details, 
                        figure_action=PLOT_ACTION, 
                        figure_path='output/'+str(id)+'/part2/figures',
                        file_name='feature_selection',
                        wspace=0.3)

    return {'pca':pca_model, 'ica':ica_model, 'srp':srp_model, 'rffs':rffs_model}


def plot_feature_srp_reconstruction(id, train_dataset, n_component_ratio_list=np.linspace(0.1, 1.0, 7)):

    recon_error_list = []
    n_components_list = [int(x*train_dataset.data.shape[1]) for x in n_component_ratio_list]
    for n_components in n_components_list:
        fs_algo = SparseRandomProjection(random_state=RANDOM_STATE, n_components=n_components)
        fs_algo.fit(train_dataset.data)
        recon_error = reconstructionError(fs_algo, train_dataset.data)
        recon_error_list.append(recon_error)
    recon_error_dict = {'x':n_components_list,'y':recon_error_list}

    print('SRP - highest reconstruction error: {:.4f}'.format(np.max(recon_error_list)))

    gen_plot(x_data=recon_error_dict['x'], 
            y_data=recon_error_dict['y'],
            title_name='Randomized Projection - Reconstruction Error',
            ylabel_name='Reconstruction Error', 
            xlabel_name='# Components',
            figure_action=PLOT_ACTION, 
            figure_path='output/'+str(id)+'/part2/figures',
            file_name='srp_reconstruction_error')
    
    return None


def etl_data(specified_dataset):
    if specified_dataset == 'spam':
        df = pd.read_csv('data/spam/spambasedata.csv', sep=',')
        print('using the dataset stored in ./data/spam')
        #shuffle data before splitting to train and test
        resampled_df = df.loc[:,:].sample(frac=1).reset_index(drop=True)
        train_frac = 0.8
        train_samples = int(round(resampled_df.shape[0]*train_frac))
        dirty_train_df = resampled_df.loc[:train_samples,:].reset_index(drop=True)
        dirty_test_df = resampled_df.loc[train_samples:,:].reset_index(drop=True)
        class_col = 'class'

    elif specified_dataset == 'aps':
        dirty_train_df = pd.read_csv('data/aps/aps_failure_training_set.csv', na_values=['na'])
        dirty_test_df = pd.read_csv('data/aps/aps_failure_test_set.csv', na_values=['na'])
        print('using the dataset stored in ./data/aps')
        class_col = 'class'

    #clean both datasets
    scaler = preprocessing.MinMaxScaler()
    train_and_test_df = clean_and_scale_dataset({'train':dirty_train_df, 'test':dirty_test_df}, scaler=scaler ,na_action=0)
    train_df, test_df = train_and_test_df[0], train_and_test_df[1]

    #prep the datasets 
    [train_dataset, test_dataset], label_encoder = prep_data({'train':train_df, 'test':test_df}, shuffle_data=True, balance_method=BALANCE_METHOD, class_col=class_col)
    print('\nTRAINING DATA INFORMATION')
    print('{} maps to {}'.format(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print('size of training dataset:', train_dataset.data.shape)
    print('class counts:\n', train_dataset.df[class_col].value_counts(), '\n')

    return train_dataset, test_dataset, label_encoder


def cluster_homogeneity_by_feature_selection(id, fs_algo_dict, cluster_algo_list, train_dataset, cluster_size_list=[2,3,4,5,7,10,15,20,25,30], n_component_ratio_list=[1.0, 0.25, 0.125, 0.0625]):

    plot_locs = [(0,0), (0,1), (0,2), (0,3)]
    cluster_model_homogeneity = {}
    
    for cluster_algo_name in cluster_algo_list:
        for i, (fs_name, fs_algo) in enumerate(fs_algo_dict.items()):
            #transform data, run clustering on all data for variations of k, 1/2 of data for variations of k, 1/4 data for variations of k, etc
            if fs_name == 'pca':
                fs_sorted_data = fs_algo.transform(train_dataset.data)
            elif fs_name == 'rffs':
                fs_algo.fit(train_dataset.data,train_dataset.target)
                feature_importances = fs_algo.feature_importances_
                fs_sorted_data = reverse_sort_by_importance(train_dataset.data, feature_importances)
            elif fs_name in ('ica', 'srp'):
                fs_data = fs_algo.transform(train_dataset.data)
                fs_sorted_data = reverse_sort_by_importance(fs_data, kurtosis(fs_data, fisher=False))
        
            n_component_variation_list = [int(x*fs_sorted_data.shape[1]) for x in n_component_ratio_list]
            iteration_data_dict = {}
            for n_components in n_component_variation_list:
                selected_data = fs_sorted_data[:, :n_components]
                #do a loop here through kmeans number of clusters
                homogeneity_list = []
                for n_clusters in cluster_size_list:
                    if cluster_algo_name == 'kmeans':
                        cluster_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
                    elif cluster_algo_name == 'em':
                        cluster_model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)

                    cluster_model.fit(selected_data)
                    cluster_model_predictions = cluster_model.predict(selected_data)
                    homogeneity_list.append(homogeneity_score(train_dataset.target,cluster_model_predictions))

                iteration_data_dict['{}-components'.format(n_components)] = {'x':cluster_size_list, 'y':homogeneity_list}
            cluster_model_homogeneity[plot_locs[i]] = {'type':'line', 
                                                    'data_dict':iteration_data_dict, 
                                                    'title':'{} with {}'.format(cluster_algo_name.upper(), fs_name.upper())}
        
        accumulate_subplots(subplot_shape=(1,4), 
                            subplot_dict=cluster_model_homogeneity, 
                            figure_action=PLOT_ACTION, 
                            flat_xlabel='# Clusters',
                            flat_ylabel='Homogeneity',
                            figure_path='output/'+str(id)+'/part3/figures',
                            file_name='{}_homogeneity_with_feature_selection'.format(cluster_algo_name.upper()))

    return None


def cluster_2d_by_feature_selection(id, fs_algo_dict, cluster_algo_list, train_dataset, selected_cluster_number, selected_component_number=2):

    plot_locs = [(0,0), (0,1), (1,0), (1,1)]
    cluster_model_clusters = {}
    
    for cluster_algo_name in cluster_algo_list:
        for i, (fs_name, fs_algo) in enumerate(fs_algo_dict.items()):
            #transform data, run clustering on all data for variations of k, 1/2 of data for variations of k, 1/4 data for variations of k, etc
            if fs_name == 'pca':
                fs_sorted_data = fs_algo.transform(train_dataset.data)
            elif fs_name == 'rffs':
                fs_algo.fit(train_dataset.data,train_dataset.target)
                feature_importances = fs_algo.feature_importances_
                fs_sorted_data = reverse_sort_by_importance(train_dataset.data, feature_importances)
            elif fs_name in ('ica', 'srp'):
                fs_data = fs_algo.transform(train_dataset.data)
                fs_sorted_data = reverse_sort_by_importance(fs_data, kurtosis(fs_data, fisher=False))

            if cluster_algo_name == 'kmeans':
                cluster_algo = KMeans(n_clusters=selected_cluster_number, random_state=RANDOM_STATE)
                cluster_algo.fit(fs_sorted_data)
                cluster_algo_predictions = cluster_algo.predict(fs_sorted_data)
                cluster_model_clusters[plot_locs[i]] = {'type':'cluster',
                                                        'model':cluster_algo,
                                                        'model_type':'kmeans',
                                                        'data':fs_sorted_data[:, :selected_component_number],
                                                        'target':train_dataset.target,
                                                        'predictions':cluster_algo_predictions,
                                                        'title':'{} with {}'.format(cluster_algo_name.upper(), fs_name.upper()),
                                                        'ylims':[-0.2, 0.2] if fs_name=='ica' else None,
                                                        'xlims':[-0.2, 0.2] if fs_name=='ica' else None}
            elif cluster_algo_name == 'em':
                cluster_algo = GaussianMixture(n_components=selected_cluster_number, random_state=RANDOM_STATE)
                cluster_algo.fit(fs_sorted_data)
                cluster_algo_predictions = cluster_algo.predict(fs_sorted_data)
                cluster_model_clusters[plot_locs[i]] = {'type':'cluster',
                                                        'model':cluster_algo,
                                                        'model_type':'em',
                                                        'data':fs_sorted_data[:, :selected_component_number],
                                                        'target':train_dataset.target,
                                                        'predictions':cluster_algo_predictions,
                                                        'title':'{} with {}'.format(cluster_algo_name.upper(), fs_name.upper()),
                                                        'ylims':[-0.2, 0.2] if fs_name=='ica' else None,
                                                        'xlims':[-0.2, 0.2] if fs_name=='ica' else None}

        accumulate_subplots(subplot_shape=(2,2), 
                            subplot_dict=cluster_model_clusters, 
                            figure_action=PLOT_ACTION,
                            figure_path='output/'+str(id)+'/part3/figures',
                            file_name='{}_clusters'.format(cluster_algo_name.upper()))
    return None 


def run_neural_network_no_feature_selection(id, nn_model, train_dataset, test_dataset, label_encoder):
    nn_model.fit(train_dataset.data, train_dataset.target)
    predictions = nn_model.predict(test_dataset.data)
    print('neural network with no feature selection - ROC-AUC: {:.2f}'.format(roc_auc_score(test_dataset.target, predictions)))

    n_iters = list([i for i in range(nn_model.n_iter_)])
    train_scores = nn_model.loss_curve_
    val_scores = nn_model.validation_scores_
    nn_details = {}
    val_curve_dict = {}
    val_curve_dict['validation'] = {'x':n_iters, 'y':val_scores}
    nn_details[0,0] = {'type':'line', 
                        'data_dict':val_curve_dict, 
                        'title':'NN Validation Curve - No Feature Selection',
                        'ylabel':'Validation Score',
                        'xlabel':'Number of Epochs',
                        'ylims':[0.5, 1.0]}

    cm = confusion_matrix(test_dataset.target, predictions)
    normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    nn_details[0,1] = {'type':'cm', 
                        'cm':cm,
                        'classes':label_encoder.classes_,
                        'title':'Confusion Matrix',
                        'normalized':False,
                        'ylabel':'True Label',
                        'xlabel':'Predicted Label'}

    nn_details[0,2] = {'type':'cm', 
                        'cm':normalized_cm,
                        'classes':label_encoder.classes_,
                        'title':'Normalized Confusion Matrix',
                        'normalized':True,
                        'ylabel':'True Label',
                        'xlabel':'Predicted Label'}

    accumulate_subplots(subplot_shape=(1,3), 
                        subplot_dict=nn_details, 
                        figure_action=PLOT_ACTION, 
                        figure_path='output/'+str(id)+'/part4/figures',
                        file_name='nn_with_no_feature_selection',
                        wspace=0.3)
    
    return None


def run_neural_network_with_feature_selection(id, nn_model, fs_algo_dict, train_dataset, test_dataset, label_encoder, n_component_list=[128,32,8]):
    plot_locs = [(0,0), (0,1), (0,2), (0,3)]

    for i, (fs_name, fs_algo) in enumerate(fs_algo_dict.items()):
        if fs_name == 'pca':
            fs_sorted_training_data = fs_algo.transform(train_dataset.data)
            fs_sorted_testing_data = fs_algo.transform(test_dataset.data)
        elif fs_name == 'rffs':
            fs_sorted_training_data = reverse_sort_by_importance(train_dataset.data, fs_algo.feature_importances_)
            fs_sorted_testing_data = reverse_sort_by_importance(test_dataset.data, fs_algo.feature_importances_)
        elif fs_name in ('ica', 'srp'):
            fs_training_data = fs_algo.transform(train_dataset.data)
            fs_testing_data = fs_algo.transform(test_dataset.data)
            fs_sorted_training_data = reverse_sort_by_importance(fs_training_data, kurtosis(fs_training_data, fisher=False))
            fs_sorted_testing_data = reverse_sort_by_importance(fs_testing_data, kurtosis(fs_training_data, fisher=False))

        nn_details = {}
        component_variation_dict = {}
        for j, n_components in enumerate(n_component_list):

            
            selected_fs_training_data = fs_sorted_training_data[:,:n_components]
            selected_fs_testing_data = fs_sorted_testing_data[:,:n_components]
            
            nn_model.fit(selected_fs_training_data, train_dataset.target)
            predictions = nn_model.predict(selected_fs_testing_data)
            print('neural network trained with {} - {}-components - ROC-AUC: {:.2f}'.format(fs_name.upper(), n_components, roc_auc_score(test_dataset.target, predictions)))

            train_scores = nn_model.loss_curve_
            val_scores = nn_model.validation_scores_
            n_iters = list([i for i in range(nn_model.n_iter_)])
            component_variation_dict['{}-components'.format(n_components)] = {'x':n_iters, 'y':val_scores}

            cm = confusion_matrix(test_dataset.target, predictions)
            normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            nn_details[plot_locs[j+1]] = {'type':'cm', 
                                        'cm':normalized_cm,
                                        'classes':label_encoder.classes_,
                                        'title':'Normalized CM - {} Components'.format(n_components),
                                        'normalized':True,
                                        'ylabel':'True Label',
                                        'xlabel':'Predicted Label'}
                                
        nn_details[plot_locs[0]] = {'type':'line', 
                                    'data_dict':component_variation_dict, 
                                    'title':'NN Validation Curve',
                                    'ylabel':'Validation Score',
                                    'xlabel':'Number of Epochs',
                                    'ylims':[0.5, 1.0]}

        accumulate_subplots(subplot_shape=(1,4), 
                            subplot_dict=nn_details, 
                            figure_action=PLOT_ACTION, 
                            figure_path='output/'+str(id)+'/part4/figures',
                            file_name='nn_with_{}'.format(fs_name.upper()),
                            wspace=0.3)
            
    return None


def kmeans_learning_curve(nn_model, train_dataset, test_dataset, cluster_type='kmeans', n_cluster_variations=[2,3,4,5,6,7,8,9,10], append_to_original_data=False):
    train_roc_auc_list = []
    test_roc_auc_list = []
    cm_list = []
    normalized_cm_list = []
    for cluster_size in n_cluster_variations:
        if cluster_type == 'kmeans':
            clusterer = KMeans(n_clusters=cluster_size, random_state=RANDOM_STATE)
            clusterer.fit(train_dataset.data)
            train_clusters = clusterer.predict(train_dataset.data).reshape(-1, 1)
            test_clusters = clusterer.predict(test_dataset.data).reshape(-1, 1)

            cluster_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
            cluster_encoder.fit(train_clusters)

            train_cluster_data = cluster_encoder.transform(train_clusters)
            test_cluster_data = cluster_encoder.transform(test_clusters)

        elif cluster_type == 'em':
            clusterer = GaussianMixture(n_components=cluster_size, random_state=RANDOM_STATE)
            clusterer.fit(train_dataset.data)
            train_cluster_data = clusterer.predict_proba(train_dataset.data)
            test_cluster_data = clusterer.predict_proba(test_dataset.data)
        
        if append_to_original_data:      
            new_train_data = np.column_stack((train_dataset.data, train_cluster_data))
            new_test_data = np.column_stack((test_dataset.data, test_cluster_data))

        else:
            new_train_data = train_cluster_data
            new_test_data = test_cluster_data

        nn_model.fit(new_train_data, train_dataset.target)
        train_predictions = nn_model.predict(new_train_data)
        test_predictions = nn_model.predict(new_test_data)

        train_roc_auc = roc_auc_score(train_dataset.target, train_predictions)
        train_roc_auc_list.append(train_roc_auc)

        test_roc_auc = roc_auc_score(test_dataset.target, test_predictions)
        test_roc_auc_list.append(test_roc_auc)
        print('neural network trained with {} - {}-clusters - ROC-AUC: {:.2f}'.format(cluster_type.upper(), cluster_size, test_roc_auc))

        cm = confusion_matrix(test_dataset.target,test_predictions)
        normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_list.append(cm)
        normalized_cm_list.append(normalized_cm)

    return n_cluster_variations, train_roc_auc_list, test_roc_auc_list, cm_list, normalized_cm_list


def run_neural_network_with_clustering(id, nn_model, clustering_algo_list, train_dataset, test_dataset, label_encoder, n_clusters_list=[2,3,4,5,6,7,8,9,10], cluster_size_for_cm=None):
    
    for clustering_algo_name in clustering_algo_list:
        cluster_details = {} 
        not_appended_cluster_list, not_appended_train_list, not_appended_test_list, not_appended_cm, not_appended_norm_cm = kmeans_learning_curve(nn_model, train_dataset, test_dataset, cluster_type=clustering_algo_name, 
                                                                                                                                                    n_cluster_variations=n_clusters_list, append_to_original_data=False)
        not_appended_curve_dict = {}
        not_appended_curve_dict['train'] = {'x':not_appended_cluster_list, 'y':not_appended_train_list}
        not_appended_curve_dict['test'] = {'x':not_appended_cluster_list, 'y':not_appended_test_list}
        cluster_details[0,0] = {'type':'line', 
                                'data_dict':not_appended_curve_dict, 
                                'title':'NN with {} - Clusters Only'.format(clustering_algo_name.upper()),
                                'ylabel':'ROC AUC Score',
                                'xlabel':'Number of Clusters',
                                'ylims':[0.5, 1.0]}
        
        if cluster_size_for_cm is not None:
            cm_index = not_appended_cluster_list[np.where(not_appended_cluster_list == cluster_size_for_cm)][0]
        else:
            cm_index = np.argmax(not_appended_test_list)

        cluster_details[0,1] = {'type':'cm', 
                                'cm':not_appended_norm_cm[cm_index],
                                'classes':label_encoder.classes_,
                                'title':'Normalized CM - {} Clusters'.format(not_appended_cluster_list[cm_index]),
                                'normalized':True,
                                'ylabel':'True Label',
                                'xlabel':'Predicted Label'}
        
        appended_cluster_list, appended_train_list, appended_test_list, appended_cm, appended_norm_cm = kmeans_learning_curve(nn_model, train_dataset, test_dataset, cluster_type=clustering_algo_name, 
                                                                                                                                n_cluster_variations=n_clusters_list, append_to_original_data=True)
                    
        appended_curve_dict = {}
        appended_curve_dict['train'] = {'x':appended_cluster_list, 'y':appended_train_list}
        appended_curve_dict['test'] = {'x':appended_cluster_list, 'y':appended_test_list}
        cluster_details[1,0] = {'type':'line', 
                            'data_dict':appended_curve_dict, 
                            'title':'NN with {} - Appended'.format(clustering_algo_name.upper()),
                            'ylabel':'ROC AUC Score',
                            'xlabel':'Number of Clusters',
                            'ylims':[0.5, 1.0]}

        if cluster_size_for_cm is not None:
            cm_index = not_appended_cluster_list[np.where(not_appended_cluster_list == cluster_size_for_cm)][0]
        else:
            cm_index = np.argmax(appended_test_list)

        cluster_details[1,1] = {'type':'cm', 
                                'cm':appended_norm_cm[cm_index],
                                'classes':label_encoder.classes_,
                                'title':'Normalized CM - {} Clusters'.format(appended_cluster_list[cm_index]),
                                'normalized':True,
                                'ylabel':'True Label',
                                'xlabel':'Predicted Label'}

        accumulate_subplots(subplot_shape=(2,2), 
                            subplot_dict=cluster_details, 
                            figure_action=PLOT_ACTION, 
                            figure_path='output/'+str(id)+'/part5/figures',
                            file_name='nn_using_{}'.format(clustering_algo_name.upper()),
                            wspace=0.3)

    return None

def main():
    batch_id = str(int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    for specified_dataset in DATASETS:
        algo_batch_id = batch_id + '-' + str(specified_dataset) #set ID for one run, so all the algos have the same ID

        #load dataset
        train_dataset, test_dataset, label_encoder = etl_data(specified_dataset)
        
        # 1. Run the clustering algorithms on the datasets and describe what you see.
        print('{} - working on part 1: Run the clustering algorithms on the datasets'.format(specified_dataset.upper()))
        cluster_details = run_cluster_variations(algo_batch_id, train_dataset, test_dataset, max_num_cluster=N_CLUSTERS)

        #part 2 - feature selection
        print('{} - working on part 2: Apply the dimensionality reduction algorithms to the two datasets'.format(specified_dataset.upper()))
        fs_models = run_feature_selection(algo_batch_id, train_dataset, test_dataset, n_rp_runs=8)
        plot_feature_srp_reconstruction(algo_batch_id, train_dataset,n_component_ratio_list=np.linspace(0.03,1,10))

        # part 3 - cluster after feature selection
        print('{} - working on part 3: Reproduce your clustering experiment on the data after youve run dimensionality reduction on it'.format(specified_dataset.upper()))
        cluster_homogeneity_by_feature_selection(algo_batch_id, fs_models, ['kmeans', 'em'], train_dataset, cluster_size_list=[2,3,4,5,7,10,15,20,25,30])
        cluster_2d_by_feature_selection(algo_batch_id, fs_models, ['kmeans', 'em'], train_dataset, selected_cluster_number=5)
                      
        if specified_dataset == 'aps':
            #only run parts 4 and 5 on APS dataset
            nn_model = MLPClassifier(hidden_layer_sizes=(100,20,), 
                                        early_stopping=True, 
                                        n_iter_no_change=50,
                                        validation_fraction=0.3,
                                        tol=0.0001, 
                                        random_state=RANDOM_STATE, 
                                        max_iter=N_EPOCHS, 
                                        learning_rate_init=0.1)

            # part 4 - neural network with feature_selection
            print('{} - working on part 4: Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 and rerun your neural network learner on the newly projected data.'.format(specified_dataset.upper()))
            run_neural_network_no_feature_selection(algo_batch_id, nn_model=nn_model, train_dataset=train_dataset, test_dataset=test_dataset, label_encoder=label_encoder)
            run_neural_network_with_feature_selection(algo_batch_id, nn_model=nn_model, fs_algo_dict=fs_models, train_dataset=train_dataset, test_dataset=test_dataset, label_encoder=label_encoder)
            
            # part 5 - neural network with clustering
            print('{} - working on part 5: Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms, treating the clusters as if they were new features'.format(specified_dataset.upper()))
            run_neural_network_with_clustering(algo_batch_id, nn_model, train_dataset=train_dataset, test_dataset=test_dataset, clustering_algo_list=['kmeans', 'em'], label_encoder=label_encoder)
    
    return None

if __name__ == '__main__': 
    main()
    