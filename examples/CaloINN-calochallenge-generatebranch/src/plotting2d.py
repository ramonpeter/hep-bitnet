import os
import argparse

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import data_util
from calc_obs import *

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
axislabelfont.set_size(24)

tickfont = FontProperties()
tickfont.set_family('serif')
tickfont.set_name('Times New Roman')
tickfont.set_size(22)

def plot_hist2d(
        file_name,
        data_x,
        data_y,
        xaxis_label=None,
        yaxis_label=None,
        xscale='linear',
        yscale='linear',
        nbins=50,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None):
    if xmin is None:
        xmin = np.min(data_x)
    if xmax is None:
        xmax = np.max(data_x)
    if ymin is None:
        ymin = np.min(data_y)
    if ymax is None:
        ymax = np.max(data_y)
    bins_x = np.linspace(xmin,xmax,nbins+1)
    bins_y = np.linspace(ymin,ymax,nbins+1)

    fig, ax = plt.subplots(1,1,figsize=(5,5))

    ax.hist2d(data_x, data_y, bins=(bins_x,bins_y), density='True', cmap='gist_heat_r')

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)

    if xaxis_label:
        ax.set_xlabel(xaxis_label, fontproperties=axislabelfont)
    if yaxis_label:
        ax.set_ylabel(yaxis_label, fontproperties=axislabelfont)

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()

def plot_all_hist2d(data_file, plot_dir, eplus=False, mask=0):
    os.makedirs(plot_dir, exist_ok=True)

    e_ratio_e_part_args = {'xaxis_label': r'\(E_{part}\)', 'yaxis_label': r'\(E_{tot}/E_{part}\)'}
    if eplus:
        e_ratio_e_part_args.update({'ymin': 0.985, 'ymax': 1.001})

    plots = [
        ('e_ratio_e_part.pdf', calc_e_parton, {}, calc_e_ratio, {}, e_ratio_e_part_args)
    ]

    for layer in [0,1,2]:
        plots.append( (f'sparsity_{layer}_e_part.pdf', calc_e_parton, {}, calc_sparsity, {'layer': layer},
            {'xaxis_label': r'\(E_{part}\)', 'yaxis_label': f'sparsity layer {layer}', 'nbins': 46, 'ymin': 0., 'ymax': 1.}) )
        plots.append( (f'e_layer_{layer}_e_part.pdf', calc_e_parton, {}, calc_e_layer_normd, {'layer': layer},
            {'xaxis_label': r'\(E_{part}\)', 'yaxis_label': f'\\(E_{layer}/E_{{tot}}\\)'}) )

    data = data_util.load_data(data_file, mask)

    for name, function1, args1, function2, args2, args_plot in plots:
        print(name)
        plot_hist2d(
            file_name=os.path.join(plot_dir, name),
            data_x=function1(data, **args1),
            data_y=function2(data, **args2),
            **args_plot
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Where to find the data')
    parser.add_argument('--plot_dir', help='Where to save the plots')
    parser.add_argument('--mask', default=0, type=int)
    args = parser.parse_args()

    plot_all_hist2d(args.file, args.plot_dir, 'eplus' in args.file, args.mask)

if __name__=='__main__':
    main()
