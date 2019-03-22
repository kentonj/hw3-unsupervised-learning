import pandas as pd 
import datetime
import time

import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import homogeneity_score

from plot_utils import plot_confusion_matrix, plot_model_family_learning_curves, plot_multi_lines, gen_plot
from etl_utils import *

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA 
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis
import scipy.sparse as sps
from scipy.linalg import pinv

#general params:
DATASETS = ('aps', 'spam') #one of ('spam', 'aps')
SAVE_CSV = True
SAVE_MODELS = False #set to true if you want to pickle save your model
PRETRAINED_MODEL_FILEPATH = None # one of (None, directory to models) - default to None to train models, if not None - looks for the specified model
RANDOM_STATE = 27
PLOT_ACTION = 'save' # (None, 'save', 'show') - default to None to avoid issues with matplotlib depending on OS
N_LC_CHUNKS = 10 #number of chunks for learning curve data segmentation
N_CV = 5 # number of kfold cross validation splits, 1/N_CV computes the validation percentage, if any
N_EPOCHS = 2000 # maximum number of epochs for neural network training
BALANCE_METHOD = 'downsample' # (int, 'downsample' or 'upsample')
SCORING_METRIC = 'roc_auc' #this works well for both balanced and imbalanced classification problems
steps_to_run = [2]

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
            clusterer = KMeans(n_clusters=cluster_size)
        elif cluster_type == 'em':
            clusterer = GaussianMixture(n_components=cluster_size)
        kmeans_models[cluster_size] = MachineLearningModel(clusterer, 
                                                        model_family=cluster_type, 
                                                        model_type='{}-Clusters'.format(cluster_size), 
                                                        framework='sklearn', id=id)
    return kmeans_models


def run_cluster_variations(id, train_dataset, test_dataset, clustering_model_list=['kmeans', 'em'], max_num_cluster=100):
    cluster_list = [x for x in range(2, max_num_cluster)]
    
    cluster_models_dict = {}
    
    for clustering_model in clustering_model_list:
        print('')
        kmeans_models = generate_clustering_algorithms(id, cluster_list, cluster_type=clustering_model)
        train_homogeneity_list = []
        test_homogeneity_list = []
        model_list = []
        for num_cluster, algo in kmeans_models.items():
            algo.model.fit(train_dataset.data)
            if algo.model_family == 'kmeans':
                algo.train_cluster_assign = algo.model.labels_
            elif algo.model_family == 'em':
                algo.train_cluster_assign = algo.model.predict(train_dataset.data)
            
            algo.train_homogeneity = homogeneity_score(train_dataset.target, algo.train_cluster_assign)
            train_homogeneity_list.append(algo.train_homogeneity)

            algo.test_cluster_assign = algo.model.predict(test_dataset.data)
            algo.test_homogeneity = homogeneity_score(test_dataset.target, algo.test_cluster_assign)
            test_homogeneity_list.append(algo.test_homogeneity)

            model_list.append(algo)
            
            print('fitting {} - number of clusters: {} - train homogeneity score: {:.3f} - test homogeneity score: {:.3f}'\
                    .format(clustering_model.upper(), num_cluster, algo.train_homogeneity, algo.test_homogeneity), 
                    end='\r',
                    flush=True)
        print('')
        cluster_models_dict[clustering_model] = {'model_list':model_list, 
                                                'cluster_list':cluster_list, 
                                                'train_homogeneity_list':train_homogeneity_list, 
                                                'test_homogeneity_list':test_homogeneity_list}
    return cluster_models_dict

def run_feature_selection(id, train_dataset, test_dataset, param_variations=None, models_to_run=['pca', 'ica', 'srp', 'rffs'], n_rp_runs=10):
    '''
    PCA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    - vary the number of principle components - see this plot: https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
    ICA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
    Randomized Projections (https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html)
    Other feature selection algorithm (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
    CONSIDER ADDING THIS TO THE DATASET OBJECT, creating a `get_pca(components=4)` or a `get_ica(components=4)` or `get_rp(components=4) or `get_dt_fs(n_components(threshold=-np.inf, max_features=4)
    '''
    if 'pca' in models_to_run:
        pca_model = PCA()
        print('working on PCA')
        pca_model.fit(train_dataset.data)
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
        pca_variance = pca_model.explained_variance_ratio_
        n_components = np.linspace(1,pca_variance.shape[0],pca_variance.shape[0])
        pca_plot_data = np.column_stack((n_components, pca_variance, cumulative_variance))


        train_dataset.pca_data = pca_model.transform(train_dataset.data)
        test_dataset.pca_data = pca_model.transform(test_dataset.data)
        # print('cumulative variance by number of components\n', pca_plot_data)
        gen_plot(x_data=pca_plot_data[:,0],
                y_data=pca_plot_data[:,1], 
                title_name='PCA - Variance by Components',
                ylabel_name='Cumulative Variance', 
                xlabel_name='Components', 
                figure_action=PLOT_ACTION, 
                figure_path=id+'/part2/figures', 
                file_name='pca_variance')

        if SAVE_CSV:
            df = pd.DataFrame(data=pca_plot_data, columns=['Components', 'Variance', 'Cumulative Variance'])
            df.to_csv(id+'/part2/pca_variance.csv', index=False, encoding='utf-8')

    if 'ica' in models_to_run:
        print('working on ICA')
        ica_model = FastICA(max_iter=100000, tol=0.001)
        # THIS THREW AN ERROR ABOUT NaNs once
        transformed_data = ica_model.fit_transform(train_dataset.data)
        
        kurtosis_score_for_all_components = kurtosis(transformed_data, fisher=False)
        rev_arg_sort_indices = np.argsort(kurtosis_score_for_all_components)[::-1]
        rev_sorted_kurtosis_score = kurtosis_score_for_all_components[rev_arg_sort_indices]

        n_components = np.linspace(1,rev_sorted_kurtosis_score.shape[0],rev_sorted_kurtosis_score.shape[0])
        ica_plot_data = np.column_stack((n_components, rev_sorted_kurtosis_score))
        
        train_dataset.ica_data = ica_model.transform(train_dataset.data)[:, rev_arg_sort_indices]
        test_dataset.ica_data = ica_model.transform(test_dataset.data)[:, rev_arg_sort_indices]

        gen_plot(x_data=ica_plot_data[:,0],
                y_data=ica_plot_data[:,1], 
                title_name='ICA - Kurtosis by Components',
                xlabel_name='Components', 
                ylabel_name='Kurtosis', 
                figure_path=id+'/part2/figures', 
                file_name='ica_kurtosis', 
                figure_action=PLOT_ACTION)
        if SAVE_CSV:
            df = pd.DataFrame(data=ica_plot_data, columns=['Components', 'Kurtosis'])
            df.to_csv(id+'/part2/ica_kurtosis.csv', index=False, encoding='utf-8')


    if 'srp' in models_to_run:
        srp_x_data_list = []
        srp_y_data_list = []
        for i in range(n_rp_runs):
            print('working on Randomized Projections - run {}'.format(i), end='\r', flush=True)
            srp_model = SparseRandomProjection(n_components=train_dataset.data.shape[1])

            transformed_data = srp_model.fit_transform(train_dataset.data)
            kurtosis_score_for_all_components = kurtosis(transformed_data, fisher=False)
            rev_sort_arg_indices = np.argsort(kurtosis_score_for_all_components)[::-1]
            rev_sorted_kurtosis_score = kurtosis_score_for_all_components[rev_sort_arg_indices]

            n_components = np.linspace(1,rev_sorted_kurtosis_score.shape[0],rev_sorted_kurtosis_score.shape[0])
            srp_plot_data = np.column_stack((n_components, rev_sorted_kurtosis_score))
            # print('reverse sorted kurtosis score:\n',srp_plot_data)
            
            if not os.path.exists(id+'/part2'):
                os.makedirs(id+'/part2')
            if SAVE_CSV:
                df = pd.DataFrame(data=srp_plot_data, columns=['Components', 'Kurtosis'])
                df.to_csv(id+'/part2/srp_kurtosis'+str(i)+'.csv', index=False, encoding='utf-8')

            srp_x_data_list.append(srp_plot_data[:,0])
            srp_y_data_list.append(srp_plot_data[:,1])
        print('')
        gen_plot(x_data=srp_x_data_list,
                y_data=srp_y_data_list, 
                multiples=True,
                title_name='SRP - Kurtosis by Components',
                xlabel_name='Components', 
                ylabel_name='Kurtosis', 
                figure_path=id+'/part2/figures', 
                file_name='srp_kurtosis', 
                figure_action=PLOT_ACTION)
        
        train_dataset.srp_data = srp_model.transform(train_dataset.data)[:, rev_arg_sort_indices]
        test_dataset.srp_data = srp_model.transform(test_dataset.data)[:, rev_arg_sort_indices]
                
    if 'rffs' in models_to_run:
        # random forest feature selection
        print('working on Random Forest feature selection')
        rffs_model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
        feature_importances = rffs_model.fit(train_dataset.data,train_dataset.target).feature_importances_ 
        rev_sort_arg_indices = np.argsort(kurtosis_score_for_all_components)[::-1]
        rev_sorted_feature_importances = feature_importances[rev_sort_arg_indices]

        n_components = np.linspace(1,rev_sorted_feature_importances.shape[0],rev_sorted_feature_importances.shape[0])
        rfs_plot_data = np.column_stack((n_components, rev_sorted_feature_importances))

        train_dataset.rffs_data = train_dataset.data[:, rev_arg_sort_indices]
        test_dataset.rffs_data = test_dataset.data[:, rev_arg_sort_indices]

        gen_plot(x_data=rfs_plot_data[:,0],
                y_data=rfs_plot_data[:,1],
                title_name='Random Forest - Feature Importance',
                xlabel_name='Components', 
                ylabel_name='Feature Importance', 
                figure_path=id+'/part2/figures', 
                file_name='rffs_feature_importance', 
                figure_action=PLOT_ACTION)

        if SAVE_CSV:
            df = pd.DataFrame(data=rfs_plot_data, columns=['Components', 'Feature Importance'])
            df.to_csv(id+'/part2/rffs_feature_importance.csv', index=False, encoding='utf-8')

    return train_dataset, test_dataset

def main():
    batch_id = str(int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    for specified_dataset in DATASETS:
        algo_batch_id = batch_id + '-' + str(specified_dataset) #set ID for one run, so all the algos have the same ID
        #load dataset
        if specified_dataset == 'spam':
            df = pd.read_csv('data/spam/spambasedata.csv', sep=',')
            print('using the dataset stored in ./data/spam')
            #shuffle data before splitting to train and test
            df = df.sample(frac=1).reset_index(drop=True)
            train_frac = 0.8
            train_samples = int(round(df.shape[0]*train_frac))
            dirty_train_df = df.iloc[:train_samples,:]
            dirty_test_df = df.iloc[train_samples:,:]
            class_col = 'class'

        elif specified_dataset == 'aps':
            dirty_train_df = pd.read_csv('data/aps/aps_failure_training_set.csv', na_values=['na'])
            dirty_test_df = pd.read_csv('data/aps/aps_failure_test_set.csv', na_values=['na'])
            print('using the dataset stored in ./data/aps')
            class_col = 'class'

        #clean both datasets
        scaler = preprocessing.MinMaxScaler()
        train_and_test_df = clean_and_scale_dataset({'train':dirty_train_df, 'test':dirty_test_df}, scaler=scaler ,na_action=-1)
        train_df, test_df = train_and_test_df[0], train_and_test_df[1]

        #prep the datasets 
        [train_dataset, test_dataset], label_encoder = prep_data({'train':train_df, 'test':test_df}, shuffle_data=True, balance_method=BALANCE_METHOD, class_col=class_col)
        print('\nTRAINING DATA INFORMATION')
        print('{} maps to {}'.format(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print('size of training dataset:', train_dataset.data.shape)
        print('class counts:\n', train_dataset.df[class_col].value_counts())

        # 1. Run the clustering algorithms on the datasets and describe what you see.
        '''
        TODO: talking points:
        - the crest of improvement, not wanting to move toward a very high number of clusters, which will end up perfectly clustering data points into clusters
        - around where the number of clusters improve - choose this as best ~10-20, this could represent different failure / non failure cases that aren't just APS/non-APS
        - for spam, could represent different bundles of spam -> some emails that are very excited sounding, some that use certain words alot, some clusters that say "password" a lot vs others that say "money" a lot
        '''
        if 1 in steps_to_run:
            cluster_details = run_cluster_variations(algo_batch_id, train_dataset, test_dataset, max_num_cluster=40)

            if PLOT_ACTION:
                plot_multi_lines(cluster_details, 
                                x_key='cluster_list', 
                                train_y_key='train_homogeneity_list', 
                                test_y_key='test_homogeneity_list', 
                                title_name='Homogeneity by Cluster Size',
                                ylabel_name='Homogeneity',
                                xlabel_name='Number of Clusters',
                                figure_action=PLOT_ACTION, 
                                figure_path=str(algo_batch_id)+'/figures',
                                file_name='clustering_models')
        '''
        TODO: 2. Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
        PCA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
        ICA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
        Randomized Projections (https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html)
        Other feature selection algorithm (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
        - I NEED TO STORE EACH OF THESE NEW REPRESENTATIONS OF THE DATA in the Dataset Object
        Talking points:
        - see if there's a way to visualize dimensionality reduction:
            - possibily plotting first two dimension of PCA, ICA, etc for each of the classes in the selected dataset (use different colors for each class in dataset)
        '''
        if 2 in steps_to_run:
            train_dataset, test_dataset = run_feature_selection(algo_batch_id, train_dataset, test_dataset)
            if not os.path.exists(algo_batch_id+'/part2/data'):
                os.makedirs(algo_batch_id+'/part2/data')
            np.savetxt(algo_batch_id+'/part2/data/train_pca_data.csv', train_dataset.pca_data, delimiter=',')
            np.savetxt(algo_batch_id+'/part2/data/train_ica_data.csv', train_dataset.ica_data, delimiter=',')
            np.savetxt(algo_batch_id+'/part2/data/train_srp_data.csv', train_dataset.srp_data, delimiter=',')
            np.savetxt(algo_batch_id+'/part2/data/train_rffs_data.csv', train_dataset.rffs_data, delimiter=',')

            np.savetxt(algo_batch_id+'/part2/data/test_pca_data.csv', test_dataset.pca_data, delimiter=',')
            np.savetxt(algo_batch_id+'/part2/data/test_ica_data.csv', test_dataset.ica_data, delimiter=',')
            np.savetxt(algo_batch_id+'/part2/data/test_srp_data.csv', test_dataset.srp_data, delimiter=',')
            np.savetxt(algo_batch_id+'/part2/data/test_rffs_data.csv', test_dataset.rffs_data, delimiter=',')


        '''
        TODO: 3. Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
        - run clustering on each of the outputs of the 2, stored in the Dataset Object
        '''

        '''
        TODO: 4. Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 
        (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
        - Use APS
        - run neural network on each variation of data from feature selection
        - compare to full dataset
        '''

        '''
        TODO: 5. Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), 
        treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. 
        Again, rerun your neural network learner on the newly projected data.
        - USE APS
        - two variations
            - one only using the clusters for Neural network
            - another appending cluster data to neural network

        - with EM, we can use probabilities too, not just predictions
        '''
        
        
    

if __name__ == '__main__': 
    main()
    