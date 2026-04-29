import pickle
import numpy as np
from neuron import h
import matplotlib.pyplot as plt
import sys

sys.path.append('../HomeostasisMotoneuronModel')
import tools.neuron_functions as nf
import tools.plotting_tools as pt
from cells import MotoneuronNoDendrites
from scripts.measure_excitability import measure_excitability


def run_simulation(learning_rates_dict, bounded_params_dict): 
    """
    Given dictionaries alpha (learning rates) and bounded parameter values, 
    return the post-homeostatic equillibrium MN firing rate and rheobase. 
    """

    h.load_file("stdrun.hoc")
    np.random.seed(0)

    mn = MotoneuronNoDendrites()

    T_total = 10000
    dt = 0.025

    num_supraspinal = 30
    rate_supraspinal_healthy = 200
    rate_supraspinal_lesion = 2
    inhomo_process = int(T_total/(6*dt))*[rate_supraspinal_healthy] + int(T_total*5/(6*dt))*[rate_supraspinal_lesion]

    supra_tau = 2
    synaptic_weight_supra = 0.00015
    shape = 1.2

    # descending input from the brain
    supraspinal_neurons = nf.create_time_series_inhomogeneous_input_neurons(num_supraspinal, inhomo_process, tStop=T_total)
    supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)
    W_supraspinal = np.random.gamma(shape, scale=synaptic_weight_supra/shape, size=[num_supraspinal, 1])
    syn_supraspinal, nc_supraspinal = nf.create_exponential_synapses(supraspinal_neurons, [mn], W_supraspinal, supra_tau)

    # recording
    cai = h.Vector().record(mn.soma(0.5).motoneuron._ref_cai)
    mn_spike_times = nf.create_spike_recorder_mns([mn])

    # targets
    target_ca = 0.00041 # based on the healthy avg Ca2+

    # learning rates
    alpha_na = learning_rates_dict['alpha_na']
    alpha_k = learning_rates_dict['alpha_k']
    alpha_ca = learning_rates_dict['alpha_ca']

    dt_homeo = 50

    t_current = 0

    h.run()

    while t_current < T_total:
        h.continuerun(t_current + dt_homeo)

        # activity signal
        ca_val = cai.to_python()[int(t_current/dt):int((t_current+dt_homeo)/dt)]
        error = target_ca - np.mean(ca_val)
        seg = mn.soma(0.5).motoneuron

        # update conductances
        gnabar = max([0, seg.gnabar + alpha_na * error])
        seg.gnabar = min([gnabar, bounded_params_dict['gnabar_max']])
        gkrect = max([0, seg.gkrect - alpha_k * error])
        seg.gkrect = max([gkrect, bounded_params_dict['gkrect_min']])
        gcaN = max([0, seg.gcaN + alpha_ca * error])
        seg.gcaN = min([gcaN, bounded_params_dict['gcaN_max']])

        t_current += dt_homeo

    mn_spike_times = mn_spike_times[0].to_python()
    mn_final_fr = len([x for x in mn_spike_times if x > (T_total-1000)])

    rheobase = measure_excitability({'gnabar': seg.gnabar, 
                          'gkrect': seg.gkrect, 
                          'gcaN': seg.gcaN})
    

    return {'rheobase': rheobase, 
            'mn_firing_rate_final': mn_final_fr}

    
if __name__ == '__main__': 
    alpha_ca = 1e-1

    bounded_params_dict = {'gnabar_max': np.inf, 
                    'gkrect_min': -np.inf, 
                    'gcaN_max': np.inf}
    
    all_rheo = []
    all_frs = []

    na_range = [x/10 for x in range(1, 100, 5)]
    k_range = [x/100 for x in range(1, 10000, 500)]

    # for na_mult in na_range: 
    #     k_rheo = []
    #     k_frs = []
    #     for k_mult in k_range: 
    #         print(f'Running simulation with: Na_mult {na_mult}, K_mult {k_mult}')

    #         learning_rates_dict = {'alpha_na': alpha_ca*na_mult,
    #                                 'alpha_k': alpha_ca*k_mult, 
    #                                 'alpha_ca': alpha_ca}
    
    #         stats = run_simulation(learning_rates_dict, bounded_params_dict)

    #         k_rheo.append(stats['rheobase'])
    #         k_frs.append(stats['mn_firing_rate_final'])
    #         print(f"Results in rheobase of: {stats['rheobase']}")
        
    #     all_rheo.append(k_rheo)
    #     all_frs.append(k_frs)
    
    # data = {}
    # data['all_rheo'] = all_rheo
    # data['all_frs'] = all_frs
    # data['na_vals'] = na_range
    # data['k_vals'] = k_range

    # f = open('results/constant_ca_no_bounds.pickle',"wb")
    # pickle.dump(data, f )
    # f.close()

    bounded_params_dict = {'gnabar_max': 0.05*1.3, 
                    'gkrect_min': 0.3*0.8, 
                    'gcaN_max': 0.05*1.04}
    
    all_rheo = []
    all_frs = []
    
    for na_mult in na_range: 
        k_rheo = []
        k_frs = []
        for k_mult in k_range: 

            print(f'Running simulation with: Na_mult {na_mult}, K_mult {k_mult}')

            learning_rates_dict = {'alpha_na': alpha_ca*na_mult,
                                    'alpha_k': alpha_ca*k_mult, 
                                    'alpha_ca': alpha_ca}
    
            stats = run_simulation(learning_rates_dict, bounded_params_dict)

            k_rheo.append(stats['rheobase'])
            k_frs.append(stats['mn_firing_rate_final'])
            print(f"Results in rheobase of: {stats['rheobase']}")
        
        all_rheo.append(k_rheo)
        all_frs.append(k_frs)
    
    data = {}
    data['all_rheo'] = all_rheo
    data['all_frs'] = all_frs
    data['na_vals'] = na_range
    data['k_vals'] = k_range

    f = open('results/constant_ca_tight_bounds.pickle',"wb")
    pickle.dump(data, f )
    f.close()

    import pdb; pdb.set_trace()
    