import os
from typing import Callable, Optional, List

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

import caloch_eval.calc_obs

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


class Plotter:

    class Histogram:

        plot_name: str
        label: str
        func:Callable[[dict], ArrayLike]
        args:dict
        n_bins:int
        x_log:bool
        y_log:bool
        range:Optional[List[float]]
        bins:Optional[ArrayLike]
        sum_hist:ArrayLike
        sq_sum_hist:ArrayLike
        train_hist: Optional[ArrayLike]
        train_hist_std: Optional[ArrayLike]
        n_updates:int

        def __init__(self,
                plot_name: str,
                label: str,
                func:Callable[[dict], ArrayLike],
                args:dict={},
                n_bins:int=50,
                x_log:bool=False,
                y_log:bool=False,
                range:Optional[List[float]]=None) -> None:
            self.plot_name = plot_name
            self.label = label
            self.func = func
            self.args = args
            self.n_bins = n_bins
            self.x_log = x_log
            self.y_log = y_log
            self.range = range
            self.bins = None

            self.sum_hist = np.zeros(n_bins)
            self.sq_sum_hist = np.zeros(n_bins)
            self.train_hist = None
            self.train_hist_std = None
            self.n_updates = 0

        def set_bins(self, observable:ArrayLike) -> None:
            if self.range is None:
                self.range = [np.min(observable), np.max(observable)]
            if self.range[0] is None:
                self.range[0] = np.min(observable)
            if self.range[1] is None:
                self.range[1] = np.max(observable)
            if not self.x_log:
                self.bins = np.linspace(self.range[0], self.range[1], self.n_bins+1)
            else:
                self.bins = np.logspace(self.range[0], self.range[1], self.n_bins+1)

        def update(self, data:dict) -> None:
            data_coppy = {k: np.copy(v) for k, v in data.items()}
            observable = self.func(data_coppy, **self.args)
            if self.x_log:
                observable = observable[observable>1e-7]
            if self.bins is None:
                self.set_bins(observable)
            histogram, _ = np.histogram(observable, bins=self.bins, density=True)
            self.sum_hist += histogram
            self.sq_sum_hist += histogram**2
            self.n_updates += 1

        def bin_train_data(self, train_data:dict) -> None:
            data_coppy = {k: np.copy(v) for k, v in train_data.items()}
            observable = self.func(data_coppy, **self.args)
            if self.x_log:
                observable = observable[observable>1e-7]
            if self.bins is None:
                self.set_bins(observable)
            self.train_hist, _ = np.histogram(observable, bins=self.bins, density=True)
            count, _ = np.histogram(observable, bins=self.bins, density=False)
            self.train_hist_std = self.train_hist/np.sqrt(count)

        def get_mean(self) -> np.ndarray:
            return self.sum_hist/self.n_updates

        def get_std(self) -> np.ndarray:
            return np.sqrt(self.sq_sum_hist/self.n_updates
                    -(self.sum_hist/self.n_updates)**2)

        def plot(self):
            FONTSIZE = 14
            labels = ['INN', 'Train']
            colors = ['#3b528b', '#1a8507']
            hists = [self.get_mean(), self.train_hist]
            hist_errors = [self.get_std(), self.train_hist_std]
            dup_last = lambda a: np.append(a, a[-1])

            fig, axs = plt.subplots(3, 1, sharex=True,
                    gridspec_kw={'height_ratios' : [4, 1, 1], 'hspace' : 0.00})

            for y, y_err, label, color in zip(hists, hist_errors, labels, colors):
                if y is None: continue

                axs[0].step(self.bins, dup_last(y), label=label, color=color,
                        linewidth=1.0, where='post')
                axs[0].step(self.bins, dup_last(y + y_err), color=color,
                        alpha=0.5, linewidth=0.5, where='post')
                axs[0].step(self.bins, dup_last(y - y_err), color=color,
                        alpha=0.5, linewidth=0.5, where='post')
                axs[0].fill_between(self.bins, dup_last(y - y_err),
                        dup_last(y + y_err), facecolor=color,
                        alpha=0.3, step='post')

                if label == 'Train': continue
                if hists[1] is None: continue

                ratio = y / hists[1]
                ratio_err = np.sqrt((y_err / y)**2 + (hist_errors[1] / hists[1])**2)
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                axs[1].step(self.bins, dup_last(ratio), linewidth=1.0, where='post', color=color)
                axs[1].step(self.bins, dup_last(ratio + ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where='post')
                axs[1].step(self.bins, dup_last(ratio - ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where='post')
                axs[1].fill_between(self.bins, dup_last(ratio - ratio_err),
                        dup_last(ratio + ratio_err), facecolor=color, alpha=0.3, step='post')

                delta = np.fabs(ratio - 1) * 100
                delta_err = ratio_err * 100

                markers, caps, bars = axs[2].errorbar((self.bins[:-1] + self.bins[1:])/2, delta,
                        yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                        linewidth=0, fmt='.', capsize=2)
                [cap.set_alpha(0.5) for cap in caps]
                [bar.set_alpha(0.5) for bar in bars]

            axs[0].legend(loc='upper right', frameon=False)
            axs[0].set_ylabel('normalized', fontsize = FONTSIZE)
            if self.y_log:
                axs[0].set_yscale('log')
            
            ylim = axs[0].get_ylim()
            if not self.y_log:
                ylim = (0.0, ylim[1])
            else:
                ylim = (max(ylim[0], 1e-5), ylim[1])
            axs[0].set_ylim(ylim)

            axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{True}}$",
                    fontsize = FONTSIZE)
            axs[1].set_yticks([0.9,1,1.1])
            axs[1].set_ylim([0.75,1.25])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

            axs[2].set_ylim((0.05,20))
            axs[2].set_yscale("log")
            axs[2].set_yticks([0.1, 1.0, 10.0])
            axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

            axs[2].axhline(y=1.0,linewidth=0.5, linestyle="--", color="grey")
            axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            axs[2].set_ylabel(r"$\delta [\%]$", fontsize = FONTSIZE)

            plt.xlabel(self.label, fontsize = FONTSIZE)
            if self.x_log:
                plt.xscale('log')
            plt.xlim(tuple(self.range))

            fig.savefig(self.plot_name, bbox_inches='tight', format='pdf', pad_inches=0.05)
            plt.close()

    histograms: List[Histogram]
    directory: str

    def __init__(self, plots:dict, directory: str) -> None:
        self.histograms = []
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        for plot in plots.keys():
            plot_param = plots[plot]
            plot_param['func'] = getattr(calc_obs, plot_param['func'])
            plot_param['plot_name'] = os.path.join(directory, plot)
            self.histograms.append(self.Histogram(**plot_param))

    def update(self, data:dict) -> None:
        for histogram in self.histograms:
            histogram.update(data)

    def bin_train_data(self, train_data:dict) -> None:
        for histogram in self.histograms:
            histogram.bin_train_data(train_data)

    def plot(self) -> None:
        for histogram in self.histograms:
            histogram.plot()
