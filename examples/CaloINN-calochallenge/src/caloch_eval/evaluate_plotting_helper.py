# pylint: disable=invalid-name
""" helper file containing plotting functions to evaluate contributions to the
    Fast Calorimeter Challenge 2022.

    by C. Krause

    Modified for the Detector Flows paper (arxiv:XXXX) by L. Favaro
"""

import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

dup = lambda a: np.append(a, a[-1])
colors = ["tab:blue", "tab:orange"]
labels = ["INN", "VAE+INN"]

plt.rc("font", family="serif", size=20)
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)
 
def plot_layer_comparison(hlf_class, data, reference_class, reference_data, arg, show=False):
    """ plots showers of of data and reference next to each other, for comparison """
    num_layer = len(reference_class.relevantLayers)
    vmax = np.max(reference_data)
    layer_boundaries = np.unique(reference_class.bin_edges)
    for idx, layer_id in enumerate(reference_class.relevantLayers):
        plt.figure(figsize=(6, 4))
        reference_data_processed = reference_data\
            [:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        reference_class._DrawSingleLayer(reference_data_processed,
                                         idx, filename=None,
                                         title='Reference Layer '+str(layer_id),
                                         fig=plt.gcf(), subplot=(1, 2, 1), vmax=vmax,
                                         colbar='None')
        data_processed = data[:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        hlf_class._DrawSingleLayer(data_processed,
                                   idx, filename=None,
                                   title='Generated Layer '+str(layer_id),
                                   fig=plt.gcf(), subplot=(1, 2, 2), vmax=vmax, colbar='both')

        filename = os.path.join(arg.output_dir,
                                'Average_Layer_{}_dataset_{}.pdf'.format(layer_id, arg.dataset))
        plt.savefig(filename, dpi=300, format='pdf')
        if show:
            plt.show()
        plt.close()

def plot_Etot_Einc_discrete(hlf_class, reference_class, arg, p_label):
    """ plots Etot normalized to Einc histograms for each Einc in ds1 """
    # hardcode boundaries?
    bins = np.linspace(0.4, 1.4, 21)
    plt.figure(figsize=(10, 10))
    target_energies = 2**np.linspace(8, 23, 16)
    for i in range(len(target_energies)-1):
        if i > 3 and 'photons' in arg.dataset:
            bins = np.linspace(0.9, 1.1, 21)
        energy = target_energies[i]
        which_showers_ref = ((reference_class.Einc.squeeze() >= target_energies[i]) & \
                             (reference_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        which_showers_hlf = ((hlf_class.Einc.squeeze() >= target_energies[i]) & \
                             (hlf_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        ax = plt.subplot(4, 4, i+1)
        counts_ref, _, _ = ax.hist(reference_class.GetEtot()[which_showers_ref] /\
                                   reference_class.Einc.squeeze()[which_showers_ref],
                                   bins=bins, label='reference', linestyle='-', density=True,
                                   histtype='stepfilled', alpha=0.2, linewidth=1.0, color=hlf_class.color)
        counts_data, _, _ = ax.hist(hlf_class.GetEtot()[which_showers_hlf] /\
                                    hlf_class.Einc.squeeze()[which_showers_hlf], bins=bins,
                                    label='generated', histtype='step', linewidth=1.5, alpha=1.,
                                    density=True, color=reference_class.color)
        if i in [0, 1, 2]:
            energy_label = 'E = {:.0f} MeV'.format(energy)
        elif i in np.arange(3, 12):
            energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
        else:
            energy_label = 'E = {:.1f} TeV'.format(energy/1e6)
        ax.text(0.95, 0.95, energy_label, ha='right', va='top',
                transform=ax.transAxes)
        ax.set_xlabel(r'$E_{\text{tot}} / E_{\text{inc}}$')
        ax.xaxis.set_label_coords(1., -0.15)
        ax.set_ylabel('counts')
        ax.yaxis.set_ticklabels([])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('Etot / Einc at E = {}: \n'.format(energy))
            f.write(str(seps))
            f.write('\n\n')
        h, l = ax.get_legend_handles_labels()
    ax = plt.subplot(4, 4, 16)
    ax.legend(h, l, loc='center', fontsize=20)
    ax.axis('off')
    filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}_E_i.pdf'.format(arg.dataset))
    plt.savefig(filename, dpi=300, format='pdf')
    plt.close()

def plot_Etot_Einc(list_hlfs, reference_class, arg, p_label):
    """ plots Etot normalized to Einc histogram """

    bins = np.linspace(0.5, 1.5, 31)
    fig, ax = plt.subplots(2,1, figsize=(5, 4.5), gridspec_kw = {"height_ratios": (4,1), "hspace": 0.0}, sharex = True)
        
    counts_ref, bins = np.histogram(reference_class.GetEtot() / reference_class.Einc.squeeze(), bins=bins, density=False)
    counts_ref_norm = counts_ref/counts_ref.sum()
    geant_error = counts_ref_norm/np.sqrt(counts_ref)
    ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                   alpha=0.8, linewidth=1.0, color='k', where='post')
    ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
    ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
    for i in range(len(list_hlfs)):
        if list_hlfs[i] == None:
            pass
        else:
            counts, _ = np.histogram(list_hlfs[i].GetEtot() / list_hlfs[i].Einc.squeeze(), bins=bins, density=False)
            counts_data, bins = np.histogram(list_hlfs[i].GetEtot() / list_hlfs[i].Einc.squeeze(), bins=bins, density=False)
            counts_data_norm = counts_data/counts_data.sum()
            ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                   linewidth=1., alpha=1., color=colors[i], linestyle='-')

            y_ref_err = counts_data_norm/np.sqrt(counts)
            ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
    
            ratio = counts_data / counts_ref
            ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
            ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', color=colors[i], alpha=0.2)

    ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(bins[0], bins[-1])

    ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
    ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
    
    ax[1].set_xlabel(r'$E_{\text{tot}} / E_{\text{inc}}$')
    ax[0].set_ylabel(r'a.u.')
    ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
    ax[0].legend(loc='best', frameon=False, title=p_label, handlelength=1.5, fontsize=15, title_fontsize=15)
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
    if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}.pdf'.format(arg.dataset))
        fig.savefig(filename, dpi=300, format='pdf')
    if arg.mode in ['all', 'hist-chi', 'hist']:
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of Etot / Einc histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('Etot / Einc: \n')
            f.write(str(seps))
            f.write('\n\n')
    plt.close()


def plot_E_layers(list_classes, reference_class, arg, p_label):
    """ plots energy deposited in each layer """
    for key in reference_class.GetElayers().keys():
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        if arg.x_scale == 'log':
            bins = np.logspace(np.log10(arg.min_energy),
                               np.log10(reference_class.GetElayers()[key].max()),
                               40)
        else:
            bins = 40
        
        counts_ref, bins = np.histogram(reference_class.GetElayers()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
        for i in range(len(list_classes)):
            if list_classes[i] == None:
                pass
            else:
                counts, _ = np.histogram(list_classes[i].GetElayers()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_classes[i].GetElayers()[key], bins=bins, density=False)
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', color=colors[i], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
 
        #ax[0].set_title("Energy deposited in layer {}".format(key))
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
        ax[1].set_xlabel(f'$E_{{{key}}}$ [MeV]')
        ax[0].set_yscale('log'), ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].legend(loc='best', frameon=False, title=p_label, handlelength=1.5, fontsize=15, title_fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir, 'E_layer_{}_dataset_{}.pdf'.format(
                key,
                arg.dataset))
            plt.savefig(filename, dpi=300, format='pdf')
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of E layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('E layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECEtas(list_hlfs, reference_class, arg, p_label):
    """ plots center of energy in eta """
    for key in reference_class.GetECEtas().keys():
        if arg.dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)

        counts_ref, bins = np.histogram(reference_class.GetECEtas()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetECEtas()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetECEtas()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', color=colors[i], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
 
        #ax[0].set_title(r"Center of Energy in $\Delta\eta$ in layer {}".format(key))
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(f'$\\langle\\eta\\rangle_{{{key}}}$ [mm]')
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5, title_fontsize=15, fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'ECEta_layer_{}_dataset_{}.pdf'.format(key,
                                                                           arg.dataset))
            plt.savefig(filename, dpi=300, format='pdf')
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of EC Eta layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('EC Eta layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECPhis(list_hlfs, reference_class, arg, p_label):
    """ plots center of energy in phi """
    for key in reference_class.GetECPhis().keys():
        if arg.dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)
        
        counts_ref, bins = np.histogram(reference_class.GetECPhis()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetECPhis()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetECPhis()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', color=colors[i], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
 
        #ax[0].set_title(r"Center of Energy in $\Delta\phi$ in layer {}".format(key))
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(f"$\\langle\\phi\\rangle_{{{key}}}$ [mm]")
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5, title_fontsize=15, fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'ECPhi_layer_{}_dataset_{}.pdf'.format(key,
                                                                           arg.dataset))
            plt.savefig(filename, dpi=300, format='pdf')
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of EC Phi layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('EC Phi layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECWidthEtas(list_hlfs, reference_class, arg, p_label):
    """ plots width of center of energy in eta """
    for key in reference_class.GetWidthEtas().keys():
        if arg.dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        fig, ax = plt.subplots(2,1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)
        
        counts_ref, bins = np.histogram(reference_class.GetWidthEtas()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetWidthEtas()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetWidthEtas()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', color=colors[i], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
        
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(r"$\sigma_{\langle\eta\rangle_{" + str(key) + "}}$ [mm]")
        #ax[0].set_title(r"Width of Center of Energy in $\Delta\eta$ in layer {}".format(key))
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5, fontsize=15, title_fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
 
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthEta_layer_{}_dataset_{}.pdf'.format(key,
                                                                              arg.dataset))
            plt.savefig(filename, dpi=300, format='pdf')
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of Width Eta layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('Width Eta layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECWidthPhis(list_hlfs, reference_class, arg, p_label):
    """ plots width of center of energy in phi """
    for key in reference_class.GetWidthPhis().keys():
        if arg.dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 51)
        
        counts_ref, bins = np.histogram(reference_class.GetWidthPhis()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None:
                pass
            else:
                counts, _ = np.histogram(list_hlfs[i].GetWidthPhis()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(list_hlfs[i].GetWidthPhis()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', color=colors[i], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
        
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(r"$\sigma_{\langle\phi\rangle_{" + str(key) + "}}$ [mm]")
        #ax[0].set_title(r"Width of Center of Energy in $\Delta\phi$ in layer {}".format(key))
        ax[0].set_xlim(*lim)
        ax[0].set_yscale('log')
        ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
        ax[0].legend(loc=(0.57, 0.54), frameon=False, title=p_label, handlelength=1.5, fontsize=15, title_fontsize=15)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
 
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthPhi_layer_{}_dataset_{}.pdf'.format(key,
                                                                              arg.dataset))
            plt.savefig(filename, dpi=300, format='pdf')
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('Width Phi layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_sparsity(list_hlfs, reference_class, arg, p_label):
    """ Plot sparsity of relevant layers"""
    for key in reference_class.GetSparsity().keys():
        lim = (0, 1)
        fig, ax = plt.subplots(2, 1, figsize=(5,4.5), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
        bins = np.linspace(*lim, 20)
        
        counts_ref, bins = np.histogram(1-reference_class.GetSparsity()[key], bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        geant_error = counts_ref_norm/np.sqrt(counts_ref)
        ax[0].step(bins, dup(counts_ref_norm), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
        ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
        ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None:
                pass
            else:
                counts, _ = np.histogram(1-list_hlfs[i].GetSparsity()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(1-list_hlfs[i].GetSparsity()[key], bins=bins, density=False)
                
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
        
                ratio = counts_data / counts_ref
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref_norm), dup(ratio+y_ref_err/counts_ref_norm), step='post', color=colors[i], alpha=0.2)

        ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[1].set_yticks((0.7, 1.0, 1.3))
        ax[1].set_ylim(0.5, 1.5)
        ax[0].set_xlim(bins[0], bins[-1])

        ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
        
        ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
        ax[0].set_ylabel(r'a.u.')
        ax[1].set_xlabel(f"$\\lambda_{{{key}}}$")
        #plt.yscale('log')
        ax[1].set_xlim(*lim)
        ax[0].legend(loc='best', frameon=False, title=p_label, handlelength=1.5, fontsize=15, title_fontsize=15)
        fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'Sparsity_layer_{}_dataset_{}.pdf'.format(key,
                                                                            arg.dataset))
            plt.savefig(filename, format='pdf')
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)), 'a') as f:
                f.write('Sparsity {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_cell_dist(list_showers, ref_shower_arr, arg, p_label):
    """ plots voxel energies across all layers """
    fig, ax = plt.subplots(2,1, figsize=(6, 6), gridspec_kw={"height_ratios": (4,1), "hspace": 0.0}, sharex=True)
    if arg.particle == 'photon':
        color = cm.gnuplot2(np.linspace(0.2, 0.8, 3)[1])
    elif arg.particle == 'pion':
        color = cm.gnuplot2(np.linspace(0.2, 0.8, 3)[2])
    else:
        color = cm.gnuplot2(np.linspace(0.2, 0.8, 3)[0])
    if arg.x_scale == 'log':
        bins = np.logspace(np.log10(arg.min_energy),
                           np.log10(ref_shower_arr.max()),
                           50)
    else:
        bins = 50

    counts_ref, bins = np.histogram(ref_shower_arr, bins=bins, density=True)
    ax[0].step(bins, dup(counts_ref), label='GEANT', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
 
    for i in range(len(list_showers)):
        counts, _ = np.histogram(list_showers[i].flatten(), bins=bins, density=False)
        counts_data, bins = np.histogram(list_showers[i].flatten(), bins=bins, density=True)
        ax[0].step(bins, dup(counts_data), label=labels[i], where='post',
                   linewidth=1.5, alpha=1., color=colors[i], linestyle='-')

        y_ref_err = counts_data/np.sqrt(counts)
        ax[0].fill_between(bins, dup(counts_data+y_ref_err), dup(counts_data-y_ref_err), step='post', color=colors[i], alpha=0.2)
    
        ratio = counts_data / counts_ref
        ax[1].step(bins, dup(ratio), linewidth=1.5, alpha=1.0, color=colors[i], where='post')
        ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref), dup(ratio+y_ref_err/counts_ref), step='post', color=colors[i], alpha=0.2)

    ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(bins[0], bins[-1])

    ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
    ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
 
    ax[0].set_title(r"Voxel energy distribution")
    ax[1].set_xlabel(r'$E$ [MeV]')
    ax[0].set_yscale('log')
    if arg.x_scale == 'log':
        ax[1].set_xscale('log')
    #plt.xlim(*lim)
    ax[0].legend(loc='best', frameon=False, title=p_label)
    fig.tight_layout()
    if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(arg.output_dir,
                                'voxel_energy_dataset_{}.pdf'.format(arg.dataset))
        plt.savefig(filename, dpi=300, format='pdf')
    if arg.mode in ['all', 'hist-chi', 'hist']:
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of voxel distribution histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir,
                               'histogram_chi2_{}.txt'.format(arg.dataset)), 'a') as f:
            f.write('Voxel distribution: \n')
            f.write(str(seps))
            f.write('\n\n')
    plt.close()

def plot_atlas_style(hlf_class, vae_class, reference_class, arg, p_label):
    """ plots histograms for all incident energies (atlas style plot)
    Also computes the Chi^2 values"""

    if arg.dataset == '1-photons':
        # p_label = r'$\gamma$ DS-1'

        bins_list = []
        for i in range(15):
            if i ==0 or i==1: bins = np.linspace(0.45, 1.3, 21)
            elif i>=2 and i<=4: bins = np.linspace(0.73, 1.1, 21)
            elif i>4 and i<= 6: bins = np.linspace(0.87,1.03,21)
            elif i>6 and i<=9: bins = np.linspace(0.935,1.01, 21)
            elif i>9: bins= np.linspace(0.935,0.998,11)

            bins_list.append(bins)

    elif arg.dataset == '1-pions':
        # p_label = r'$\pi^{+}$ DS-1'

        bins_list = []
        for i in range(15):
            if i >=0 and i<4: bins = np.linspace(0, 1.5, 21)
            elif i>=4 and i< 7: bins = np.linspace(0.4, 1.2, 21)
            elif i>=7 and i< 12: bins = np.linspace(0.6,1.1,21)
            elif i>=12: bins= np.linspace(0.7,1.1,11)

            bins_list.append(bins)

    elif arg.dataset == '2':       
        raise ValueError("No discrete incident energies for dataset 2")
        # p_label = r'$e^{+}$ DS-2'

    else:
        raise ValueError("No discrete incident energies for dataset 3")
        # p_label = r'$e^{+}$ DS-3'

    hlf_class_2 = vae_class    

    color = colors[0]
    color2 = colors[1]
    color_true = 'k'

    fig, ax = plt.subplots(8, 5, figsize=(15,13.5), gridspec_kw={'height_ratios': [6,2,1,6,2,1,6,2], 'wspace':0, 'hspace':0.0})

    # "-1" since the VAE energies are a little bit to low due to rounding errors. Since we are checking for
    # imtervalls, it is no problem.
    target_energies = 2**np.linspace(8, 23, 16) - 1
    total_chi2 = 0
    total_chi2_2 = 0
    total_bins = 0
    even_pairs = list(product((0,3,6), (0,1,2,3,4)))
    odd_pairs = list(product((1,4,7),(0,1,2,3,4)))
    fake_pairs = list(product((2,5),(0,1,2,3,4)))
    for i in fake_pairs:
        ax[i].remove()
    for i in range(len(target_energies)-1):

        bins=bins_list[i]

        total_bins += np.size(bins) -1
        energy = target_energies[i]
        even_pair = even_pairs[i]
        odd_pair = odd_pairs[i]
        which_showers_ref = ((reference_class.Einc.squeeze() >= target_energies[i]) & \
                            (reference_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        which_showers_hlf = ((hlf_class.Einc.squeeze() >= target_energies[i]) & \
                            (hlf_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        which_showers_hlf_2 = ((hlf_class_2.Einc.squeeze() >= target_energies[i]) & \
                            (hlf_class_2.Einc.squeeze() < target_energies[i+1])).squeeze()

        
        energy_ref = reference_class.GetEtot()[which_showers_ref] / reference_class.Einc.squeeze()[which_showers_ref]
        energy_data_1 = hlf_class.GetEtot()[which_showers_hlf] / hlf_class.Einc.squeeze()[which_showers_hlf] 
        energy_data_2 = hlf_class_2.GetEtot()[which_showers_hlf_2] / hlf_class_2.Einc.squeeze()[which_showers_hlf_2],

        counts_ref, bins = np.histogram(energy_ref, bins=bins, density=False)
        counts_ref_norm = counts_ref/counts_ref.sum()
        ref_error = counts_ref_norm/np.sqrt(counts_ref)

        counts_data_1, bins = np.histogram(energy_data_1, bins=bins, density=False)
        counts_data_1_norm = counts_data_1/counts_data_1.sum()
        data_1_error = counts_data_1_norm/np.sqrt(counts_data_1)
         
        counts_data_2, bins = np.histogram(energy_data_2, bins=bins, density=False)
        counts_data_2_norm = counts_data_2/counts_data_2.sum()
        data_2_error = counts_data_2_norm/np.sqrt(counts_data_2)
         
        if i in [0, 1, 2]:
            energy_label = 'E = {:.0f} MeV'.format(energy)
        elif i in np.arange(3, 12):
            energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
        else:
            energy_label = 'E = {:.1f} TeV'.format(energy/1e6)

        ax[even_pair].step(bins, dup(counts_ref_norm), color=color_true,
                           label=energy_label, alpha=0.8, linewidth=1., linestyle='-', where='post')
        ax[even_pair].step(bins, dup(counts_data_1_norm), color=color,
                                linewidth=1., alpha=1., where='post')
        ax[even_pair].step(bins, dup(counts_data_2_norm), color=color2,
                                linewidth=1., alpha=1., where='post')
        
        ax[even_pair].fill_between(bins, dup(counts_ref_norm+ref_error), dup(counts_ref_norm-ref_error), step='post', color=color_true, alpha=0.2)
        ax[even_pair].fill_between(bins, dup(counts_data_1_norm+data_1_error), dup(counts_data_1_norm-data_1_error), step='post', color=color, alpha=0.2)
        ax[even_pair].fill_between(bins, dup(counts_data_2_norm+data_2_error), dup(counts_data_2_norm-data_2_error), step='post', color=color2, alpha=0.2)
    
        ratio_data_1 = counts_data_1_norm / counts_ref_norm
        ratio_data_2 = counts_data_2_norm / counts_ref_norm

        ax[odd_pair].step(bins, dup(ratio_data_1), linewidth=1.0, alpha=1.0, color=color, where='post')
        ax[odd_pair].step(bins, dup(ratio_data_2), linewidth=1.0, alpha=1.0, color=color2, where='post')
        ax[odd_pair].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
        ax[odd_pair].axhline(0.7, c='k', ls='--', lw=0.5)
        ax[odd_pair].axhline(1.3, c='k', ls='--', lw=0.5)

        ax[odd_pair].fill_between(bins, dup(1-ref_error/counts_ref_norm), dup(1+ref_error/counts_ref_norm), step='post', color=color_true, alpha=0.2 )
        ax[odd_pair].fill_between(bins, dup(ratio_data_1-data_1_error/counts_ref_norm), dup(ratio_data_1+data_1_error/counts_ref_norm), step='post', color=color, alpha=0.2)
        ax[odd_pair].fill_between(bins, dup(ratio_data_2-data_2_error/counts_ref_norm), dup(ratio_data_2+data_1_error/counts_ref_norm), step='post', color=color2, alpha=0.2)

        total_counts_ref = len(energy_ref)
        total_counts_data = len(energy_data_1)
        total_counts_data_2 = len(energy_data_2)

        energy = energy+1
        
        #ax[even_pair].tick_params(axis='both', which='major', labelsize=8, direction='in', pad=-10)
        #ax[odd_pair].tick_params(axis='both', which='major', labelsize=8, direction='in', pad=-5)
        ax[even_pair].set_xlim(bins[0], bins[-1])
        ax[odd_pair].set_xlim(bins[0], bins[-1])
        ax[odd_pair].set_ylim(0.5, 1.5)
        ax[odd_pair].set_yticks((0.7, 1.3))

        #ax[even_pair].set_yticks(ax[even_pair].get_yticks()[::2])
        ax[even_pair].set_yticks([])
        ax[even_pair].set_xticks([])
        #ax[odd_pair].set_xticks([])
        ax[odd_pair].set_yticks([])

        if odd_pair[1]==0:
            #ax[even_pair].set_yticks(ax[even_pair].get_yticks()[::2])
            ax[odd_pair].set_ylabel(r'$\frac{\text{Model}}{\text{GEANT}}$')
            ax[even_pair].set_ylabel('a.u.')
            ax[odd_pair].set_yticks((0.7, 1.3))

        if odd_pair[0]==7:
            ax[odd_pair].set_xlabel(f'$E_{{\\text{{tot}}}} / E_{{\\text{{inc}}}}$')

        if i in [0, 1, 2]:
            energy_label = 'E = {:.0f} MeV'.format(energy)
        elif i in np.arange(3, 12):
            energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
        else:
            energy_label = 'E = {:.1f} TeV'.format(energy/1e6)

        ax[even_pair].text(0.03, 0.9, energy_label, fontsize=16, transform=ax[even_pair].transAxes)
        #seps = _separation_power(counts_ref, counts_data, bins)
        #seps_2 = _separation_power(counts_ref, counts_data_2, bins)
        #chi2 = chi2_eval(counts_ref, counts_data, total_counts_ref, total_counts_data, bins)
        #chi2_2 = chi2_eval(counts_ref, counts_data_2, total_counts_ref, total_counts_data_2, bins)

        #total_chi2 += chi2
        #total_chi2_2 += chi2_2
        #print("INN: Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps))
        #print("VAE+INN: Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps_2))
        #print("INN: Chi2 of Etot / Einc at E = {} histogram: {}".format(energy, chi2))
        #print("VAE+INN: Chi2 of Etot / Einc at E = {} histogram: {}".format(energy, chi2_2))

    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    #print("INN: Total chi2 of Etot / Einc histograms: {}".format(total_chi2))
    #print("VAE+INN: Total chi2 of Etot / Einc histograms: {}".format(total_chi2_2))
    #print("Total bins: ", total_bins)
    #print("INN: Total chi2 per dof of Etot / Einc histograms: {}".format(total_chi2/total_bins))
    #print("VAE+INN: Total chi2 per dof of Etot / Einc histograms: {}".format(total_chi2_2/total_bins))

    
    #ax = plt.subplot(4, 4, 16)
    #ax.legend(h, l, loc='center', fontsize =13, frameon=False)
    #ax.axis('off')
    #plt.text(0.05, 0.1, r'$\chi^2_{\text{INN}}$/NDF ' + r'$={:.2f}$'.format(total_chi2/total_bins), fontsize = 13)
    #plt.text(0.05, -0.1, r'$\chi^2_{\text{VAE+INN}}$/NDF ' + r'$={:.2f}$'.format(total_chi2_2/total_bins), fontsize = 13)
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.955, 0.955))
    
    line1, = ax[0,0].plot(0,0, c='tab:blue', label='solid')
    line2, = ax[0,0].plot(0,0, c='tab:orange', ls='solid', label='dashed')
    line3, = ax[0,0].plot(0,0, c='k', ls='solid')
    fig.legend(handles=[line1,line2,line3], labels=['INN', 'VAE+INN', 'GEANT'], ncol=3, frameon=False, loc='upper center')

    filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}_E_i.pdf'.format(arg.dataset))
    fig.savefig(filename, dpi=300)
    # plt.savefig('gamma_discrete_450_slides.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()

def chi2_eval(hist1, hist2, total_counts_ref, total_counts_data, bins):
    #hist1 is normalized ref counts, hist2 is normalized data counts
    #total_counts_ref is the total number of reference counts in histo
    #total_counts_data is the total number of data counts in histo
    ret = (hist1 - hist2)**2
    hist1_unnorm, hist2_unnorm = hist1*np.diff(bins)*total_counts_ref, hist2*np.diff(bins)*total_counts_data
    sigma_sq = hist1_unnorm/((np.diff(bins)*total_counts_ref)**2)+hist2_unnorm/((np.diff(bins)*total_counts_data)**2)
    ret /= sigma_sq
    return np.nansum(ret)
