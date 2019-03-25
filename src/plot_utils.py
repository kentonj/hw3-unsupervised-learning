import platform
import matplotlib as mpl
if platform.mac_ver()[0] != '':
    print('mac os version detected:', platform.mac_ver()[0], ' - switching matplotlib backend to TkAgg')
    mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from scipy import linalg
DEFAULT_COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

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

# THIS IS THE ONE TO USE FOR MULTIPLE SUBPLOTS
def accumulate_subplots(subplot_shape, subplot_dict, big_title=None, flat_xlabel=None, flat_ylabel=None, 
                        outer_xlabel_list=None, outer_ylabel_list=None, label_outside=False,
                        figure_action='show', figure_path='figures/lc', file_name=None, sharex=False, sharey=False, figure_size_multiplier=None, **kwargs):

    if figure_size_multiplier is None:
        figure_size_multiplier = 5

    f, axarr = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=(figure_size_multiplier*subplot_shape[1], figure_size_multiplier*subplot_shape[0]), sharex=sharex, sharey=sharey)

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
        elif subplot_type == 'cm':
            make_cm_subplot(subplot=specified_subplot, **subplot_payload)

    hspace = kwargs.get('hspace',0.3)
    wspace = kwargs.get('wspace',0.2)

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

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

def make_ellipses(subplot, gmm, color_cycle):
    for n in range(gmm.weights_.shape[0]):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                    180 + angle, color=color_cycle[n])
        ell.set_clip_box(subplot.bbox)
        ell.set_alpha(0.5)
        subplot.add_artist(ell)
        subplot.set_aspect('equal', 'datalim')

def make_cluster_subplot(subplot, data, target, predictions, model_type, model, scale_axes=False, **kwargs):
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
        cluster_colors = [DEFAULT_COLOR_CYCLE[x] for x in predictions_with_label]
        subplot.scatter(rows_with_label[:, 0], rows_with_label[:, 1], c=cluster_colors, s=25, marker=marker_variations[i], alpha=0.3)
        i += 1
    # show the centers
    if model_type == 'kmeans':
        center_colors = [DEFAULT_COLOR_CYCLE[x] for x in range(model.cluster_centers_.shape[0])]
        subplot.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c=center_colors, s=200, alpha=0.95, edgecolors='red')
    elif model_type == 'em':
        center_colors = [DEFAULT_COLOR_CYCLE[x] for x in range(model.weights_.shape[0])]
        make_ellipses(subplot, model, color_cycle=center_colors)
    if scale_axes:
        subplot.set_xlim([-np.mean(data[:, 0])*100, np.mean(data[:, 0])*100])
        subplot.set_ylim([-np.mean(data[:, 1])*100, np.mean(data[:, 1])*100])

    ylims = kwargs.get('ylims',None)
    xlims = kwargs.get('xlims',None)

    if ylims is not None:
        subplot.set_ylim(ylims)
    if xlims is not None:
        subplot.set_xlim(xlims)

    
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
    default_line_type = '-'

    for i, (legend_name, values) in enumerate(data_dict.items()):
        x_values = values.get('x',None)
        y_values = values.get('y',None)
        line_type = values.get('line_type',None)
        line_color = values.get('line_color',None)

        if line_type is None:
            line_type = default_line_type
        
        if line_color is None:
            if x_values is not None:
                subplot.plot(x_values, y_values, label=legend_name, linestyle=line_type, color=DEFAULT_COLOR_CYCLE[i])
            else:
                subplot.plot(y_values, label=legend_name, linestyle=line_type, color=DEFAULT_COLOR_CYCLE[i])
        else:
            if x_values is not None:
                subplot.plot(x_values, y_values, label=legend_name, linestyle=line_type, color=DEFAULT_COLOR_CYCLE[i])
            else:
                subplot.plot(y_values, label=legend_name, linestyle=line_type, color=DEFAULT_COLOR_CYCLE[i])

    title = kwargs.get('title', None)
    ylabel = kwargs.get('ylabel', None)
    xlabel = kwargs.get('xlabel', None)
    ylims = kwargs.get('ylims', None)
    xlims = kwargs.get('xlims', None)
    if title is not None:
        subplot.set_title(title)
    if ylabel is not None:
        subplot.set_ylabel(ylabel)
    if xlabel is not None:
        subplot.set_xlabel(xlabel)
    if ylims is not None:
        subplot.set_ylim(bottom=min(ylims), top=max(ylims))
    if xlims is not None:
        subplot.set_xlim(bottom=min(xlims), top=max(xlims))
    
    subplot.legend(loc='best')
        
def make_cm_subplot(subplot, cm, classes, normalized=False, cmap=plt.cm.Blues, **kwargs):
    tick_marks = np.arange(len(classes))
    plt.setp(subplot, xticks=tick_marks, xticklabels=classes, yticks=tick_marks, yticklabels=classes)


    subplot.imshow(cm, interpolation='nearest', cmap=cmap)
    thresh = cm.max() / 2.

    if normalized:
        string_formatting = '.2f'
    else:
        string_formatting = '.0f'

    for j, k in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        subplot.text(k, j, format(cm[j, k], string_formatting),
                        horizontalalignment="center",
                        color="white" if cm[j, k] > thresh else "black")


    title = kwargs.get('title', None)
    ylabel = kwargs.get('ylabel', None)
    xlabel = kwargs.get('xlabel', None)
    if title is not None:
        subplot.set_title(title)
    if ylabel is not None:
        subplot.set_ylabel(ylabel)
    if xlabel is not None:
        subplot.set_xlabel(xlabel)