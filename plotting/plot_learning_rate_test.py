import pickle
from typing import List
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_heatmap(data, 
                 xticks: List = None, 
                 yticks: List = None,
                 xlabel: str = '', 
                 ylabel: str = '',
                 cmap: str = 'coolwarm', 
                 cbar_label: str = '',
                 ax = None): 
    if ax is None: 
        plt.figure()
        ax = plt.gca()

    if cmap == 'coolwarm': 
        norm = colors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
        im = ax.imshow(data, cmap, norm=norm)
    else: 
        im = ax.imshow(data, cmap, vmax=10)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if xticks != None: 
        ax.set_xticks([x for x in range(0,len(xticks),3)], xticks[0:len(xticks):3], fontsize=10)
        ax.set_yticks([x for x in range(0,len(yticks),3)], yticks[0:len(yticks):3], fontsize=10)
    
    plt.colorbar(im, label=cbar_label)
    
    return ax
    


if __name__ == '__main__': 
    baseline_rheo_val = 0.095
    file_no_bounds = "results/constant_ca_no_bounds.pickle"
    with open(file_no_bounds, "rb") as f:
        data = pickle.load(f)
    data_rheo = np.array(data['all_rheo'])
    data_rheo_norm = (data_rheo-baseline_rheo_val)/baseline_rheo_val*100
    ax = plot_heatmap(data_rheo_norm, 
                      cbar_label='Percent change from baseline rheobase (%)',
                      xticks=data['na_vals'], 
                      xlabel='Na+ conductance (scale of g_Ca2+)', 
                      yticks=data['k_vals'], 
                      ylabel='K+ conductance (scale of g_Ca2+)')
    ax.set_title('Relative rheobase post-homeostasic process \n(constant Ca2+)', fontsize=10)

    plt.savefig('plotting/plots/relative_rheobase_constant_ca_heatmap.png', dpi=300)


    ax = plot_heatmap(data['all_frs'], 
                      cbar_label='Motoneuron firing rate (Hz)',
                      xticks=data['na_vals'], 
                      xlabel='Na+ conductance (scale of g_Ca2+)', 
                      yticks=data['k_vals'], 
                      ylabel='K+ conductance (scale of g_Ca2+)', 
                      cmap='Purples')
    ax.set_title('Motoneuron firing post-homeostasic process \n(constant Ca2+)', fontsize=10)

    plt.savefig('plotting/plots/mn_firing_constant_ca_heatmap.png', dpi=300)
    
    file_bounds = "results/constant_ca_tight_bounds.pickle"
    with open(file_bounds, "rb") as f:
        data = pickle.load(f)
    data_rheo = np.array(data['all_rheo'])
    data_rheo_norm = (data_rheo-baseline_rheo_val)/baseline_rheo_val*100
    ax = plot_heatmap(data_rheo_norm, 
                      cbar_label='Percent change from baseline rheobase (%)',
                      xticks=data['na_vals'], 
                      xlabel='Na+ conductance (scale of g_Ca2+)', 
                      yticks=data['k_vals'], 
                      ylabel='K+ conductance (scale of g_Ca2+)')
    ax.set_title('Relative rheobase post-homeostasic process \n(constant Ca2+, bounded conductances)', fontsize=10)

    plt.savefig('plotting/plots/relative_rheobase_constant_ca_heatmap_bounded_conductances.png', dpi=300)

    ax = plot_heatmap(data['all_frs'], 
                      cbar_label='Motoneuron firing rate (Hz)',
                      xticks=data['na_vals'], 
                      xlabel='Na+ conductance (scale of g_Ca2+)', 
                      yticks=data['k_vals'], 
                      ylabel='K+ conductance (scale of g_Ca2+)', 
                      cmap='Purples')
    ax.set_title('Motoneuron firing post-homeostasic process \n(constant Ca2+, bounded conductances)', fontsize=10)
    
    plt.savefig('plotting/plots/mn_firing_constant_ca_heatmap_bounded_conductances.png', dpi=300)

    plt.show()
