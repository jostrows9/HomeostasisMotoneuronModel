"""
Plot the average calcium levels as a function of supraspinal excitation.
"""

import numpy as np
from neuron import h
import matplotlib.pyplot as plt
import sys

sys.path.append('../HomeostasisMotoneuronModel')
import tools.neuron_functions as nf
import tools.plotting_tools as pt
from cells import MotoneuronNoDendrites

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def get_avg_calcium(supra_firing_rate: int):
    h.load_file("stdrun.hoc")
    np.random.seed(0)

    mn = MotoneuronNoDendrites()

    num_supraspinal = 30
    supra_tau = 2
    synaptic_weight_supra = 0.00015
    shape = 1.2

    # descending input from the brain
    supraspinal_neurons = nf.create_input_neurons(num_supraspinal, rate=supra_firing_rate, noise=1)
    supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)
    W_supraspinal = np.random.gamma(shape, scale=synaptic_weight_supra/shape, size=[num_supraspinal, 1])
    syn_supraspinal, nc_supraspinal = nf.create_exponential_synapses(supraspinal_neurons, [mn], W_supraspinal, supra_tau)

    # recording
    mn_spike_times = nf.create_spike_recorder_mns([mn])
    cai = h.Vector().record(mn.soma(0.5).motoneuron._ref_cai)

    h.finitialize()
    h.tstop = 1000
    h.run()


    return {'avg_ca': np.mean(cai.to_python())*1000,
            'mn_fr': len(mn_spike_times[0].to_python())}

if __name__ == '__main__': 
    supra_firing_rates = range(100, 1000, 10)
    seeds = range(0, 5)

    mn_frs = []
    avg_cas = []
    for rate in supra_firing_rates: 
        avg_ca_dict = get_avg_calcium(rate)
        mn_frs.append(avg_ca_dict['mn_fr'])
        avg_cas.append(avg_ca_dict['avg_ca'])

    plt.subplots(2, 1, figsize=(5,6))
    plt.subplot(2, 1, 1)
    plt.plot([x*30/1000 for x in supra_firing_rates], avg_cas, ".", c='grey', markersize=10)
    plt.xlabel('Total supraspinal firing rate (kHz)', fontsize=10)
    plt.ylabel('Average internal \n calcium (uA/cm^2)', fontsize=10)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplot(2, 1, 2)
    plt.plot(mn_frs, avg_cas,  ".", c='grey', markersize=10)
    plt.xlabel('Average motoneuron firing rate (Hz)', fontsize=10)
    plt.ylabel('Average internal \n calcium (uA/cm^2)', fontsize=10)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig('/Users/juliaostrowski/Documents/DocumentsiCloud/spring26/mathneuro/calcium_as_function_excitation.png', dpi=300)

    plt.show()