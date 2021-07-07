import pandas as pd
import src.util.data_handler as dh
from src.util.const import *
from src.util.naming import *
import math

import matplotlib
matplotlib.use(MATPLOTLIB_BACKEND)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="ticks", palette=PALETTE)
C_MAP = dict(zip(STRATEGY_LABEL.values(), sns.color_palette(PALETTE, 5)))


def make_fig_single(x_name, y_name, data, title, p_type='scatter', save=False, path='', logy=False, logx=False,
                    strat='alls'):
    name = f'{strat}_{y_name}_{x_name}'
    fig, ax = plt.subplots()

    if p_type == 'scatter':
        name = f"{name}_s_single.{PLOT_FORMAT}"

        sns.scatterplot(data=data, x=x_name, y=y_name, hue="Strategy", palette=C_MAP)

    elif p_type == 'box':
        name = f"{name}_b_single.{PLOT_FORMAT}"

        sns.boxplot(x=x_name, y=y_name, hue='Strategy', data=data, ax=ax, palette=C_MAP)

    elif p_type == 'lm':
        name = f"{name}_l_single.{PLOT_FORMAT}"

        sns.lmplot(data=data, x=x_name, y=y_name, hue="Strategy", palette=C_MAP, x_ci="sd")

    elif p_type == 'cdf_c':

        name = f"cov_cdf.{PLOT_FORMAT}"

        d_min = data[PLOT_Y_LABEL['cover']].min()
        d_max = data[PLOT_Y_LABEL['cover']].max()

        sns.ecdfplot(data=data, x=PLOT_Y_LABEL['cover'], hue="Strategy", ax=ax)
        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)

    elif p_type == 'cdf_cd':

        name = f"cov_d_cdf.{PLOT_FORMAT}"

        d_min = data['Cover Difference'].min()
        d_max = data['Cover Difference'].max()

        sns.ecdfplot(data=data, x=f"Cover Difference", hue="Strategies", ax=ax)
        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)
        ax.axvline(c='r', linestyle='--', alpha=0.3)

    elif p_type == 'cdf_a':

        name = f"alloc_cdf.{PLOT_FORMAT}"

        d_min = data[PLOT_Y_LABEL['alloc']].min()
        d_max = data[PLOT_Y_LABEL['alloc']].max()
        print(data)
        sns.ecdfplot(data=data, x=PLOT_Y_LABEL['alloc'], hue="Strategy", ax=ax)
        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)

    elif p_type == 'cdf_ad':

        name = f"alloc_d_cdf.{PLOT_FORMAT}"

        print('Getting min/max')
        d_min = data['Allocation Difference [Gbps]'].min()
        d_max = data['Allocation Difference [Gbps]'].max()
        med = data['Allocation Difference [Gbps]'].median()
        mean = data['Allocation Difference [Gbps]'].mean()

        sns.ecdfplot(data=data, x=PLOT_Y_LABEL['alloc_d'], hue="Strategies", ax=ax)

        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)
        ax.axvline(x=med, c='b', linestyle='--', alpha=0.3, label='Median')
        ax.axvline(x=mean, c='y', linestyle='--', alpha=0.3, label='Mean')

    elif p_type == 'cdf_adr':

        name = f"alloc_rat_cdf.{PLOT_FORMAT}"

        print('Getting min/max')
        d_min = data['Allocation Ratio'].min()
        d_max = data['Allocation Ratio'].max()
        med = data['Allocation Ratio'].median()
        mean = data['Allocation Ratio'].mean()

        sns.ecdfplot(data=data, x='Allocation Ratio', hue="Strategies", ax=ax)

        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)
        #ax.axvline(x=med, c='b', linestyle='--', alpha=0.3, label='Median')
        #ax.axvline(x=mean, c='m', linestyle='--', alpha=0.3, label='Mean')
        ax.axvline(x=1, c='r', linestyle='--', alpha=0.3, label='One')

    elif p_type == 'cdf_ar':

        name = f"alloc_r_cdf.{PLOT_FORMAT}"

        d_min = data[PLOT_Y_LABEL['alloc']].min()
        d_max = data[PLOT_Y_LABEL['alloc']].max()

        sns.ecdfplot(data=data, x=PLOT_Y_LABEL['alloc'], hue="Ratio", ax=ax)
        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)

    elif p_type == 'cdf_cr':

        name = f"cov_r_cdf.{PLOT_FORMAT}"

        d_min = data[PLOT_Y_LABEL['cover']].min()
        d_max = data[PLOT_Y_LABEL['cover']].max()

        sns.ecdfplot(data=data, x=PLOT_Y_LABEL['cover'], hue="Ratio", ax=ax)
        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)

    elif p_type == 'cdf_d':
        name = f"deg_cdf.{PLOT_FORMAT}"

        d_min = np.min(data['degrees'])
        d_max = np.max(data['degrees'])

        sns.ecdfplot(data=data, ax=ax)
        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)

    elif p_type == 'cdf_ct':
        name = f"cov_t_cdf.{PLOT_FORMAT}"

        d_min = data[PLOT_Y_LABEL['cover']].min()
        d_max = data[PLOT_Y_LABEL['cover']].max()

        sns.ecdfplot(data=data, x=PLOT_Y_LABEL['cover'], hue="Cover Threshold", ax=ax, palette=sns.color_palette('Accent', 5))
        ax.set_xlim(d_min, d_max)
        ax.grid(b=True)

    else:
        raise ValueError(f'Error: Unsupported Plot Type: {p_type}')

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')


    fig.set_size_inches(18.5, 10.5, forward=True)
    #h, l = ax.get_legend_handles_labels()
    #ax.get_legend().remove()
    #lgd = fig.legend(handles=h, labels=l)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        print(f'saving under: {os.path.join(path, name)}')
        plt.savefig(os.path.join(path, name))#, bbox_extra_artists=(lgd,),  bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_fig_split(x_name, y_name, data, title, strategies, p_type='scatter', save=False, path='', logy=False,
                   logx=False):

    name = f'{y_name}_{x_name}'

    rs = math.ceil(math.sqrt(len(strategies)))
    fig, axs = plt.subplots(rs, rs, sharey=True)   # , sharey=True)
    axs = axs.flatten()

    if p_type == 'scatter':
        name = f"{name}_s_split.{PLOT_FORMAT}"

        i = 0
        for s in strategies:
            sns.scatterplot(data=data[(data["Strategy"] == STRATEGY_LABEL[s])], x=x_name, y=y_name, hue="Strategy",
                            ax=axs[i], palette=C_MAP)

            i = i + 1

    elif p_type == 'box':
        name = f"{name}_b_split.{PLOT_FORMAT}"

        i = 0
        print(f"Strat Column: {data['Strategy']}")
        for s in strategies:
            #print(f'Strat Label: {s}')
            print(f"Strat Column selected: {data[data['Strategy'] == STRATEGY_LABEL[s]]}")
            print(f"Data: {data[(data['Strategy'] == s)][['Allocations Gbps', 'Strategy']]}")
            sns.boxplot(data=data[(data["Strategy"] == s)], x=x_name, y=y_name, hue="Strategy",
                            ax=axs[i], palette=C_MAP)
            i = i + 1

    elif p_type == 'lm':
        name = f"{name}_l_split.{PLOT_FORMAT}"

        i = 0
        for s in strategies:
            if s in ['sqos_ob', 'sqos_ot']:
                sns.lmplot(data=data[(data["Strategy"] == STRATEGY_LABEL[s])], x=x_name, y=y_name, hue="Strategy",
                           ax=axs[i], palette=C_MAP)
            i = i + 1

    elif p_type == 'cdf_ct':
        name = f"cov_t_cdf_s.{PLOT_FORMAT}"

        d_min = data[PLOT_Y_LABEL['cover']].min()
        d_max = data[PLOT_Y_LABEL['cover']].max()

        i = 0
        for s in strategies:
            sns.ecdfplot(data=data[(data["Strategy"] == STRATEGY_LABEL[s])], x=PLOT_Y_LABEL['cover'], hue="Cover Threshold", ax=axs[i], palette=sns.color_palette('Accent', 5))
            axs[i].set_xlim(d_min, d_max)
            axs[i].grid(b=True)
            axs[i].set_title([STRATEGY_LABEL[s]])
            i = i + 1

    # TODO: Implement
    elif p_type == 'hm':
        name = f'heatmap.{PLOT_FORMAT}'

        i = 0
        for s in strategies:

            sc = axs[i].scatter(data['Source Degree'], data['Destination Degree'], c=data[PLOT_Y_LABEL['alloc']], cmap='copper')
            fig.colorbar(sc, ax=axs[i])
            i = i + 1

    else:
        raise ValueError(f'Error: Unknown Plot Type in Plot Handler: {p_type}')

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')

    fig.set_size_inches(18.5, 10.5, forward=True)

    i = 0
    hs = []
    ls = []
    for s in strategies:
        h, l = axs[i].get_legend_handles_labels()
        hs.extend(h)
        ls.extend(l)
        #axs[i].get_legend().remove()
        i = i+1

    lgd = fig.legend(handles=hs, labels=ls)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig(os.path.join(path, name), bbox_extra_artists=(lgd,))    # ,  bbox_inches='tight')
        plt.close()
    else:
        plt.show()
