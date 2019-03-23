import platform
import matplotlib as mpl
if platform.mac_ver()[0] != '':
    print('mac os version detected:', platform.mac_ver()[0], ' - switching matplotlib backend to TkAgg')
    mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

def plot_confusion_matrix(algo_family, algo_list, classes, cmap=plt.cm.Blues, figure_action='show', figure_path='figures/cm', file_name=None):
    '''
    adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    '''
    f, axarr = plt.subplots(2, len(algo_list))
    tick_marks = np.arange(len(classes))
    plt.setp(axarr, xticks=tick_marks, xticklabels=classes, yticks=tick_marks, yticklabels=classes)
    for i in range(len(algo_list)):
        # i = column of subplot
        algo = algo_list[i]
        axarr[0, i].imshow(algo.get_cm(), interpolation='nearest', cmap=cmap)
        axarr[0, i].set_title(str(algo.model_type))
        
        thresh = algo.get_cm().max() / 2.
        for j, k in itertools.product(range(algo.get_cm().shape[0]), range(algo.get_cm().shape[1])):
            axarr[0, i].text(k, j, format(algo.get_cm()[j, k], 'd'),
                            horizontalalignment="center",
                            color="white" if algo.get_cm()[j, k] > thresh else "black")


        axarr[1, i].imshow(algo.get_normalized_cm(), interpolation='nearest', cmap=cmap)
        
        thresh = algo.get_normalized_cm().max() / 2.
        for j, k in itertools.product(range(algo.get_normalized_cm().shape[0]), range(algo.get_normalized_cm().shape[1])):
            axarr[1, i].text(k, j, format(algo.get_normalized_cm()[j, k], '.2f'),
                            horizontalalignment="center",
                            color="white" if algo.get_normalized_cm()[j, k] > thresh else "black")

    for ax in axarr.flat:
        ax.set(xlabel='Predicted label', ylabel='True label')
    for ax in axarr.flat:
        ax.label_outer()
    plt.tight_layout()
    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/'+str(algo.model_family)+'.png')
    plt.close()
    return None

def plot_model_family_learning_curves(model_family, algo_list, figure_action='show', figure_path='figures/lc', file_name=None):
    line_type_dict = {
        'train':'-',
        'validation':'-.'
    }
    color_list = ['b','g','r','c','m','y','k','w', 'orange']
    
    plt.figure()
    plt.title('Learning Curves - ' + model_family)
    plt.ylabel('Score')
    if model_family == 'NeuralNetwork':
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Training Samples')

    for i in range(len(algo_list)):
        algo = algo_list[i]
        line_color = color_list[i]

        plt.plot(algo.train_sizes, 
                algo.get_train_scores(), 
                line_type_dict['train'], 
                color=line_color,
                label=(algo.model_type+' Training Score'))
        plt.plot(algo.train_sizes, 
                algo.get_validation_scores(), 
                line_type_dict['validation'], 
                color=line_color,
                label=(algo.model_type+' Validation Score'))
    plt.legend(loc='best')
    plt.grid()

    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/'+str(algo.model_family)+'.png')
    plt.close()
    return None

def plot_multi_lines(data_dict, x_key, train_y_key, test_y_key, title_name='Clustering Algorithms', ylabel_name='Homogeneity', xlabel_name='Clusters', figure_action='show', figure_path='figures/lc', file_name=None):
    line_type_dict = {
        'train':'-',
        'test':'-.'
    }
    color_list = ['b','g','r','c','m','y','k','w','orange']
    
    plt.figure()
    plt.title(title_name)
    plt.ylabel(ylabel_name)
    plt.xlabel(xlabel_name)
    i = 0
    for model_name, model_values in data_dict.items():
        
        line_color = color_list[i]

        plt.plot(model_values[x_key], 
                model_values[train_y_key], 
                line_type_dict['train'], 
                color=line_color,
                label=(model_name.upper()+' Training Homogeneity'))
        plt.plot(model_values[x_key], 
                model_values[test_y_key], 
                line_type_dict['test'], 
                color=line_color,
                label=(model_name.upper()+' Testing Homogeneity'))
        i += 1
    plt.legend(loc='best')
    plt.grid()

    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/plot.png')
    plt.close()
    return None

def gen_plot(y_data, x_data=None, multiples=False, title_name=None, ylabel_name=None, xlabel_name=None, figure_action='show', figure_path='figures/lc', file_name=None):
    plt.figure()
    
    if multiples:
        for i in range(len(y_data)):
            if x_data is not None:
                selected_y_data = y_data[i]
                selected_x_data = x_data[i]
                plt.plot(selected_x_data, selected_y_data)
            else:
                selected_y_data = y_data[i]
                plt.plot(selected_y_data)
    else:
        if x_data is not None:
            plt.plot(x_data, y_data)
        else:
            plt.plot(y_data)
    
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)

    if title_name==None:
        title_name = ylabel_name + ' by ' + xlabel_name

    plt.title(title_name)

    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/plot.png')
    plt.close()
    plt.show()


def plot_gmm(X, Y_, means, covariances, index, title):
    '''
    adapted from: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html 

    use:
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')
    '''
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)

def plot_clusters(data, target, predictions, cluster_centers):
    '''
    group by class 
    scatter the data for each class
    '''
    marker_variations = ['x', 'o']
    unique_labels = np.unique(target)
    # print(unique_labels)
    i = 0 #iterator for choosing label
    for label in unique_labels:
        row_filter = target==label
        rows_with_label = data[row_filter, :]
        predictions_with_label = predictions[row_filter]
        plt.scatter(rows_with_label[:, 0], rows_with_label[:, 1], c=predictions_with_label, s=25, marker=marker_variations[i], alpha=0.3)
        i += 1

    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c=[x for x in range(cluster_centers.shape[0])], s=200, alpha=0.95, edgecolors='red');
    plt.show()

def make_cluster_plot(data, target, predictions, cluster_centers):
    '''
    group by class 
    scatter the data for each class
    '''
    marker_variations = ['x', 'o']
    unique_labels = np.unique(target)
    i = 0 #iterator for choosing label
    for label in unique_labels:
        row_filter = target==label
        rows_with_label = data[row_filter, :]
        predictions_with_label = predictions[row_filter]
        plt.scatter(rows_with_label[:, 0], rows_with_label[:, 1], c=predictions_with_label, s=25, marker=marker_variations[i], alpha=0.3)
        i += 1

    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c=[x for x in range(cluster_centers.shape[0])], s=200, alpha=0.95, edgecolors='red')






# THIS IS THE ONE TO USE FOR MULTIPLE SUBPLOTS
def accumulate_subplots(subplot_shape, subplot_dict, big_title=None, flat_xlabel=None, flat_ylabel=None, 
                        outer_xlabel_list=None, outer_ylabel_list=None, label_outside=False,
                        figure_action='show', figure_path='figures/lc', file_name=None):
    '''
    pass in 
    '''
    f, axarr = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=(5*subplot_shape[1], 5*subplot_shape[0]))

    for (subplot_i, subplot_j), subplot_payload in subplot_dict.items():
        if subplot_shape[0] == 1:
            specified_subplot = axarr[subplot_j]
        elif subplot_shape[1] == 1:
            specified_subplot = axarr[subplot_i]  
        else:
            specified_subplot = axarr[subplot_i,subplot_j]

        subplot_type = subplot_payload['type']
        if subplot_type == 'cluster':
            make_cluster_subplot(subplot=specified_subplot,**subplot_payload)
        elif subplot_type == 'line':
            make_line_subplot(subplot=specified_subplot,**subplot_payload)

    plt.subplots_adjust(hspace=0.4)


    if big_title is not None:
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}
        f.text(0.5, 0.975, big_title, horizontalalignment='center', verticalalignment='top', fontdict=font)


    if flat_xlabel is not None:
        for ax in axarr.flat:
            ax.set(xlabel=flat_xlabel)
    # elif outer_xlabel_list is not None:
    #     for i in range(axarr.shape[1]):
    #         ax = axarr[]
    #         ax.set(xlabel=outer_xlabel_list[i])

    if flat_ylabel is not None:
        for ax in axarr.flat:
            ax.set(ylabel=flat_ylabel) 
    # elif outer_ylabel_list is not None:
    #     for i in range(axarr.shape[0]):
    #         ax.set(ylabel=outer_ylabel_list[i])

    if label_outside:
        for ax in axarr.flat:
            ax.label_outer()
    

    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/'+str(algo.model_family)+'.png')
    plt.close()
    return None

def make_cluster_subplot(subplot, data, target, predictions, cluster_centers, scale_axes=False, **kwargs):
    '''
    group by class 
    scatter the data for each class
    '''
    marker_variations = ['x', 'o']
    unique_labels = np.unique(target)
    i = 0 #iterator for choosing label
    for label in unique_labels:
        print('making cluster subplot')
        row_filter = target==label
        rows_with_label = data[row_filter, :]
        predictions_with_label = predictions[row_filter]
        subplot.scatter(rows_with_label[:, 0], rows_with_label[:, 1], c=predictions_with_label, s=25, marker=marker_variations[i], alpha=0.3)
        i += 1
    # show the centers
    subplot.scatter(cluster_centers[:,0], cluster_centers[:,1], c=[x for x in range(cluster_centers.shape[0])], s=200, alpha=0.95, edgecolors='red')

    if scale_axes:
        subplot.set_xlim([-np.mean(data[:, 0])*100, np.mean(data[:, 0])*100])
        subplot.set_ylim([-np.mean(data[:, 1])*100, np.mean(data[:, 1])*100])

    title = kwargs.get('title', None)
    ylabel = kwargs.get('ylabel', None)
    xlabel = kwargs.get('xlabel', None)
    if title is not None:
        subplot.set_title(title)
    if ylabel is not None:
        subplot.set_ylabel(ylabel)
    if xlabel is not None:
        subplot.set_xlabel(xlabel)

def make_line_subplot(subplot, data_dict, **kwargs):
    '''
    by default takes a dictionary like this:
    data_dict = {
        'line1':{'x':[1,2,3,4,5], 'y':[2,4,6,8,10]},
        'line2':{'x':[6,7,8,9,10], 'y':[12,14,16,18,20]}
    }
    '''
    print('making multiline subplot')
    
    for legend_name, values in data_dict.items():
        print('working on line: {}'.format(legend_name))
        x_values = values.get('x',None)
        y_values = values.get('y',None)
        if x_values is not None:
            subplot.plot(x_values, y_values, label=legend_name)
        else:
            subplot.plot(y_values, label=legend_name)

    title = kwargs.get('title', None)
    ylabel = kwargs.get('ylabel', None)
    xlabel = kwargs.get('xlabel', None)
    if title is not None:
        subplot.set_title(title)
    if ylabel is not None:
        subplot.set_ylabel(ylabel)
    if xlabel is not None:
        subplot.set_xlabel(xlabel)
    
    subplot.legend(loc='best')
        

