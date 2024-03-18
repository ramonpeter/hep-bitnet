import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math
import torch

import pandas as pd
from matplotlib import cm
# from matplotlib.transforms import Bbox

import data_util
from caloch_eval.calc_obs import *
from caloch_eval.evaluate_plotting_helper import *

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['text.usetex'] = True

labelfont = FontProperties()
labelfont.set_family('serif')
labelfont.set_name('Times New Roman')
labelfont.set_size(20)

axislabelfont = FontProperties()
axislabelfont.set_family('serif')
axislabelfont.set_name('Times New Roman')
axislabelfont.set_size(20)

tickfont = FontProperties()
tickfont.set_family('serif')
tickfont.set_name('Times New Roman')
tickfont.set_size(20)



def plot_average_table(data, save_file):
    
    number_of_runs = len(data)
    row_indices = [f"Run {i+1}" for i in range(number_of_runs)]

    row_indices.append("mean")
    row_indices.append("std")

    averaged = np.array([np.mean(data, axis=0), np.std(data, axis=0)])
    data = np.concatenate([data, averaged])

    df = pd.DataFrame(data, columns=["Accuracy", "AUC", "JSD", "best epoch"], index=row_indices)

    number_of_cols = len(df.columns)
    number_of_rows = len(list(df.iterrows()))
    row_labels = np.array(list(df.iterrows()), dtype=object)[:,0]
    fig, ax = plt.subplots(figsize=((number_of_rows+1), (number_of_cols)))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=df.values.round(4), colLabels=df.columns, rowLabels=row_labels,
            loc='center', cellLoc="left",colWidths = [0.1]*len(df.columns),
            colLoc="center", rowLoc="center")
    fig.tight_layout()

    plt.savefig(save_file)
    plt.close()

def plot_hist(
        file_name,
        data,
        reference,
        p_ref='photon',
        axis_label=None,
        xscale='linear',
        yscale='log',
        vmin=None,
        vmax=None,
        n_bins=100,
        ymin=None,
        ymax=None,
        ax=None,
        panel_ax=None,
        panel_scale="linear",
        density=True):
    data = data[np.isfinite(data)]
    reference = reference[np.isfinite(reference)]

    if vmin is None:
        vmin = min(np.min(data), np.min(reference))
    if vmax is None:
        vmax = max(np.max(data), np.max(reference))
    if xscale=='log':
        if vmin==0:
            vmin = min(np.min(data[data>1e-7]), np.min(reference[reference>1e-7]))
        if isinstance(n_bins, int):
            bins = np.logspace(np.log10(vmin), np.log10(vmax), n_bins)
        else:
            bins = n_bins
    else:
        if isinstance(n_bins, int):
            bins = np.linspace(vmin, vmax, n_bins)
        else:
            bins = n_bins
    
    colors = cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    if p_ref == 'electron':
        color = colors[0]
    elif p_ref == 'photon':
        color = colors[1]
    elif p_ref == 'pion':
        color = colors[2]
    else:
        color = 'blue'
        
    create_fig = False
    if ax is None:
        create_fig = True
        fig, ax = plt.subplots(1,1,figsize=(6,6))

    ns_0, bins_0, patches_0 = ax.hist(data, bins=bins, histtype='step', linewidth=2,
        alpha=1, color=color, density=density, label='VAE')

    ns_1, bins_1, patches_1 = ax.hist(reference, bins=bins, histtype='stepfilled',
            alpha=0.5, color=color, density=density, label='GEANT')

    if panel_ax is not None:
        assert len(bins_0) == len(bins_1)
        #assert (bins_0 - bins_1 < 1.e-7).all()
        
        # prevent divisions by 0! Set these bars to 0
        mask = ns_1 == 0
        ns_1[mask] = 1
        panel_data = ns_0/ns_1
        
        panel_data[mask] = float("nan")
        
        widths = 1.2*(bins_1[1:] - bins_1[:-1])
        
        panel_ax.bar(bins_0[:-1], ns_0/ns_1, label='VAE/GEANT', width=widths)
        
        panel_ax.plot([vmin, vmax],[1,1], color="red", ls="--", marker=None)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if panel_ax is not None:
        panel_ax.set_yscale(panel_scale)
        panel_ax.set_xscale(xscale)

    ax.set_xlim([vmin,vmax])
    if panel_ax is not None:
        panel_ax.set_xlim([vmin,vmax])
        panel_ax.set_ylim([0.5, 1.5])
        
    if ymin is not None or ymax is not None:
        ax.set_ylim((ymin, ymax))

    if panel_ax is not None:
        panel_ax.legend()

    if axis_label:
        if panel_ax is None:
            ax.set_xlabel(axis_label, fontproperties=axislabelfont)
        else:
            panel_ax.set_xlabel(axis_label, fontproperties=axislabelfont)
            
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    if create_fig:
        fig.tight_layout()
        fig.savefig(file_name, bbox_inches='tight')
        
    if panel_ax is None:
        plt.close()
    
def plot_loss(
        file_name,
        loss_train,
        loss_test,
        skip_epochs=True):
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    c = len(loss_test)/len(loss_train)
    ax.plot(c*np.arange(1,len(loss_train)+1), loss_train, color='blue', label='train loss')
    ax.plot(np.arange(1,len(loss_test)+1), loss_test, color='red', label='test loss')
    ax.legend(loc='upper right', prop=labelfont)

    ax.set_xlim([0,len(loss_test)])
    # nested np.mins needed for the case of different length
    # print(len(loss_test))
    if len(loss_test) <= 10 or (not skip_epochs):
        y_min = np.min(np.min(np.array([loss_train, loss_test], dtype=object)))
        y_max = np.max(np.max(np.array([loss_train, loss_test], dtype=object)))
    elif len(loss_test) <= 20:
        train_idx = 10 * len(loss_train) // len(loss_test)
        y_min = np.min(np.min(np.array([loss_train[train_idx:], loss_test[10:]], dtype=object)))
        y_max = np.max(np.max(np.array([loss_train[train_idx:], loss_test[10:]], dtype=object)))
    # elif len(loss_test) <= 20:
    else:
        train_idx = 20 * len(loss_train) // len(loss_test)
        y_min = np.min(np.min(np.array([loss_train[train_idx:], loss_test[20:]], dtype=object)))
        y_max = np.max(np.max(np.array([loss_train[train_idx:], loss_test[20:]], dtype=object)))
        
    # print(y_min, y_max)
    if y_min > 0:
        if y_max > 0:
            ax.set_ylim([y_min*0.9, y_max*1.1])
        else:
            ax.set_ylim([y_min*0.9, y_max*0.9])
    else:
        if y_max > 0:
            ax.set_ylim([y_min*1.1, y_max*1.1])
        else:
            ax.set_ylim([y_min*1.1, y_max*0.9])
            
    ax.set_xlabel('epoch', fontproperties=axislabelfont)
    ax.set_ylabel('loss', fontproperties=axislabelfont)

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()

def plot_lr(
        file_name,
        learning_rate,
        batches_per_epoch=1):
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    ax.plot(np.arange(1,len(learning_rate)+1)/batches_per_epoch, learning_rate, color='red', label='learning rate')

    ax.set_xlim([0,len(learning_rate)/batches_per_epoch])
    ax.set_xlabel('epoch', fontproperties=axislabelfont)
    ax.set_ylabel('learning rate', fontproperties=axislabelfont)

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()
    
def plot_grad(
        file_name,
        gradients,
        batches_per_epoch=1):
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    ax.plot(np.arange(1,len(gradients)+1)/batches_per_epoch, gradients, color='red', label='gradient')

    ax.set_xlim([0,len(gradients)/batches_per_epoch])
    ax.set_xlabel('epoch', fontproperties=axislabelfont)
    ax.set_ylabel('gradient', fontproperties=axislabelfont)
    ax.set_yscale("log")

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()
    
def plot_logsig(
        file_name,
        logsigs):
    
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    colors = ["red", "blue", "green", "orange"]
    labels = ["max", "min", "mean", "median"]
    for logsig, label, color in zip(logsigs, labels, colors):
        
        ax.plot(logsig, label=label, color=color)

        ax.set_xlim([0,len(logsig)])
        ax.set_xlabel('epoch', fontproperties=axislabelfont)
        ax.set_ylabel('$log(\\sigma^2)$', fontproperties=axislabelfont)


    ax.legend()
    fig.tight_layout()
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()
 
def get_all_plot_parameters(hlf, params):
    plots = []

    particle_type = params.get("particle_type", "pion")
    min_energy = params.get("min_energy", 10)

    # total energy plots
    plots.append((Etot_Einc, 'Etot_Einc.pdf', {},
                {"axis_label": r'$E_{tot} / E_{inc}$', "p_ref": particle_type,
                "vmin": 0.5, "vmax": 1.5, "yscale": "linear"}))

    # layer energy plots
    for layer in hlf.GetElayers().keys():
        plots.append((E_layers, f'E_layer_{layer}.pdf', {"layer": layer},
                    {"axis_label": f"Energy deposited in layer {layer} [MeV]", "p_ref": particle_type,
                    "vmin": min_energy, 'xscale': 'log', "n_bins": 40}))
        
    # eta centroid plots mean
    for layer in hlf.GetECEtas().keys():
        # TODO: different for dataset 2,3
        if layer in [12, 13]:
            vmin, vmax = (-500., 500.)
        else:
            vmin, vmax = (-100., 100.)
            
        plots.append((ECEtas, f'ECEta_layer_{layer}.pdf', {"layer": layer},
                    {"axis_label": r"Center of Energy in $\Delta\eta$ in layer" + f" {layer} [mm]", 
                    "p_ref": particle_type, "vmin": vmin, "vmax": vmax}))
        
    # phi centroid plots mean
    for layer in hlf.GetECPhis().keys():
        # TODO: different for dataset 2,3
        if layer in [12, 13]:
            vmin, vmax = (-500., 500.)
        else:
            vmin, vmax = (-100., 100.)
            
        plots.append((ECEtas, f'ECPhi_layer_{layer}.pdf', {"layer": layer},
                    {"axis_label": r"Center of Energy in $\Delta\phi$ in layer" + f" {layer} [mm]", 
                    "p_ref": particle_type, "vmin": vmin, "vmax": vmax}))
        
    # eta centroid plots width
    for layer in hlf.GetWidthEtas().keys():
        # TODO: different for dataset 2,3
        if layer in [12, 13]:
            vmin, vmax = (0., 400.)
        else:
            vmin, vmax = (0., 100.)
            
        plots.append((ECWidthEtas, f'WidthECEta_layer_{layer}.pdf', {"layer": layer},
                    {"axis_label": f"Width of Center of Energy in \n$\\Delta\\eta$ in layer {layer} [mm]", 
                    "p_ref": particle_type, "vmin": vmin, "vmax": vmax}))

    # phi centroid plots width
    for layer in hlf.GetWidthPhis().keys():
        # TODO: different for dataset 2,3
        if layer in [12, 13]:
            vmin, vmax = (0., 400.)
        else:
            vmin, vmax = (0., 100.)

        plots.append((ECWidthPhis, f'WidthECEta_layer_{layer}.pdf', {"layer": layer},
            {"axis_label": f"Width of Center of Energy in \n$\\Delta\\phi$ in layer {layer} [mm]", 
                "p_ref": particle_type, "vmin": vmin, "vmax": vmax}))    

    # All voxel energies flattened
    plots.append((Etot_Einc, 'Etot_Einc.pdf', {},
        {"axis_label": r'Voxel energy distribution', "p_ref": particle_type, "xscale": "log"}))

    # Sparsity
    for layer in hlf.GetSparsity().keys():
        plots.append((sparsity, f'Sparsity_layer_{layer}.pdf', {"layer": layer},
            {"axis_label": f"Sparsity of layer {layer}", "p_ref": particle_type, "yscale": "linear", 
                'n_bins': np.linspace(0, 1, 20), 'vmin': -0.05, 'vmax': 1.05}))

    return plots

def plot_all_hist(results_dir, gen_data, ref_data, params, layer_boundaries, threshold=1.e-4,
        single_plots=True, summary_plot=False, epoch=None):

    if epoch:
        plot_dir = os.path.join(results_dir, 'plots', f'epoch_{epoch:03d}')
    else:
        plot_dir = os.path.join(results_dir, 'plots/final')
    os.makedirs(plot_dir, exist_ok=True)

    xml_p = params.get('xml_ptype')
    xml_file = params.get('xml_path')
    gen_data = data_util.load_data_calo(results_dir+'/samples.hdf5', layer_boundaries)
    ref_data = data_util.load_data(params['data_path'], xml_p, xml_file)[0]

    # Load the hlf classes
    hlf_true = data_util.get_hlf(ref_data, params["xml_ptype"], layer_boundaries, threshold=threshold)
    hlf_fake = data_util.get_hlf(gen_data, params["xml_ptype"], layer_boundaries, threshold=threshold)

    os.makedirs(plot_dir, exist_ok=True)

    plots = get_all_plot_parameters(hlf_true, params)

    # Plot every histrogramn in its own file
    if single_plots:
        for function, name, args1, args2 in plots:
            plot_hist(
                    file_name=os.path.join(plot_dir, name),
                    data=function(hlf_fake, **args1),
                    reference=function(hlf_true, **args1),
                    **args2
                    )

    # Plot all the histogramms in one file
    if summary_plot:
        number_of_plots = len(plots)
        rows = number_of_plots // 6
        if number_of_plots%6 != 0:
            rows += 1
        heights = [1, 0.3, 0.3]*rows

        fig, axs = plt.subplots(rows*3,6, dpi=500, figsize=(6*7,6*np.sum(heights)), gridspec_kw={'height_ratios': heights})

        iteration = 0
        for i in range(rows*3):

            if i%3 == 1:
                iteration -= 6

            for j in range(6):

                if i % 3 == 2:
                    # Add one (small) invisible plot as whitespace
                    axs[i,j].set_visible(False)
                    continue

                elif iteration >= number_of_plots:
                    # Plots are empty remove them
                        axs[i,j].set_visible(False)
                        iteration += 1
                        continue


                # Select the correct plot input for this axis
                function, name, args1, args2 = plots[iteration]


                if i % 3 == 0:
                    # plot the main data
                    plot_hist(
                            file_name=None,
                            data=function(hlf_fake, **args1),
                            reference=function(hlf_true, **args1),
                            ax=axs[i,j],
                            panel_ax=axs[i+1,j],
                            **args2)

                    # Hide the (shared) x-axis
                    axs[i,j].xaxis.set_visible(False)

                    # Hide the first tick label
                    plt.setp(axs[i,j].get_yticklabels()[0], visible=False)  
                    iteration += 1

                if i % 3 == 1:
                    plt.setp(axs[i,j].get_yticklabels()[-1], visible=False)
                    iteration += 1             

        fig.subplots_adjust(hspace=0)
        # fig.savefig(os.path.join(os.path.join(plot_dir,"../"), "final.pdf"), bbox_inches='tight', dpi=500)
        fig.savefig(plot_dir+"/summary.pdf", bbox_inches='tight', dpi=500)
        # Dont use tight_layout!
        plt.close() 

def plot_all_hist_old(x_true, c_true, x_fake, c_fake, params, layer_boundaries, plot_dir, threshold=1.e-4):
    
    os.makedirs(plot_dir, exist_ok=False)
    
    def get_args_for_plotting(params, plot_dir):
        """Returns a args element for the plotting"""
        # TODO: Needs update for dataset 2 & 3
        
        parser_replacement = {"dataset" : params["particle_type"] + "s",
                            "output_dir" : plot_dir,
                            "mode" : "all",
                            "x_scale": "log",
                            "min_energy": 10}
        
        
        args = argparse.Namespace(**parser_replacement)
        
        args.min_energy = {'photons': 10, 'pions': 10,
                        '2': 0.5e-3/0.033, '3': 0.5e-3/0.033}[args.dataset]
        
        
        return args

    def plot(hlf_true, hlf_fake, args):
        plot_Etot_Einc(hlf_fake, hlf_true, args)
        plot_E_layers(hlf_fake, hlf_true, args)
        plot_ECEtas(hlf_fake, hlf_true, args)
        plot_ECPhis(hlf_fake, hlf_true, args)
        plot_ECWidthEtas(hlf_fake, hlf_true, args)
        plot_ECWidthPhis(hlf_fake, hlf_true, args)
        # plot_cell_dist(hlf_fake, hlf_true, args)
        if args.dataset[0] == '1':
            plot_Etot_Einc_discrete(hlf_fake, hlf_true, args)
            
    args = get_args_for_plotting(params, plot_dir=plot_dir)
    hlf_true = utils.get_hlf(x_true, c_true, params["particle_type"], layer_boundaries, threshold=threshold)
    hlf_fake = utils.get_hlf(x_fake, c_fake, params["particle_type"], layer_boundaries, threshold=threshold)
    
    plot(hlf_true, hlf_fake, args)
          
def plot_latent(samples, results_dir, epoch=None):
    if epoch is not None:
        plot_dir = os.path.join(results_dir, 'latent', f'epoch_{epoch:03d}')
    else:
        plot_dir = os.path.join(results_dir, 'latent')
    os.makedirs(plot_dir, exist_ok=True)
    
    max_dims = samples.shape[1]
    
    # previously:
    # latent_dims = [1, 150, 300, 400, 500, 504, 505, 506]
    np.linspace(0, max_dims-4, 5)
    
    # Cover the space equally and look at the extra dims dimensions
    latent_dims = list(np.linspace(0, max_dims-1, 5).astype(int))
        
    
    for idx in latent_dims:
        min_v = -3
        max_v = 3
        bins = np.linspace(min_v, max_v, 51)

        fig,axs = plt.subplots(1,1,figsize=(6,4))

        axs.hist(samples[:,idx], bins, color='red', histtype='step', density=True)

        axs.set_xlim(min_v, max_v)

        z_ = np.linspace(min_v, max_v, 501)
        p = 1/np.sqrt(2*np.pi)*np.exp(-z_**2/2)
        axs.plot(z_, p, color='black')

        plt.xticks(fontproperties=tickfont)
        plt.yticks(fontproperties=tickfont)
        axs.set_yscale('log')
        axs.set_xlabel(f'\\(z_{{{idx+1}}}\\)', fontproperties=axislabelfont)
        axs.set_ylabel('normalized distribution', fontproperties=axislabelfont)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'latent_{idx:03d}.pdf'), bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='Where to find the results and save the plots')
    parser.add_argument('--reference_file', help='Where to find the reference data')
    parser.add_argument('--include_coro', action='store_true', help='Also plot the pixel to pixel correlation (Computationally expensive)')
    parser.add_argument('--layer', default=None, type=int, help='Which layer to plot')
    args = parser.parse_args()

    # plot_all_hist(args.results_dir, args.reference_file, args.include_coro, args.layer)

if __name__=='__main__':
    main()
