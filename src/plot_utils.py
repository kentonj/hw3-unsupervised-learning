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

