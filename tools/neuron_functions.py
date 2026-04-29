from neuron import h
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from numpy.random import choice


def create_input_neurons(N, rate, noise, first_spike=0):
    supraspinal_neurons = []
    if type(rate) == np.ndarray:
        for i in range(N):
            if rate[i]==0:
                cell = h.NetStim()
                cell.interval = 1000.0    # Inter-spike interval in ms
                cell.noise = noise
                cell.number = 1e999
                cell.start = 1e999
                supraspinal_neurons.append(cell)
            else:
                cell = h.NetStim()
                cell.interval = 1000.0/ rate[i]   # Inter-spike interval in ms
                cell.noise = noise
                cell.number = 1e999
                cell.start = first_spike
                supraspinal_neurons.append(cell)

    else:
        for _ in range(N):
            cell = h.NetStim()
            cell.interval = 1000.0 / rate  # Inter-spike interval in ms
            cell.noise = noise
            cell.number = 1e999
            cell.start = first_spike
            supraspinal_neurons.append(cell)
    return supraspinal_neurons


def create_spike_recorder_input_neurons(neurons):
    num_neurons = len(neurons)
    spike_times = [h.Vector() for _ in range(num_neurons)]
    spike_detector = [h.NetCon(neurons[i], None) for i in range(num_neurons)]
    for i in range(num_neurons): spike_detector[i].record(spike_times[i])
    return spike_times


def create_spike_recorder_mns(neurons):
    MN_spike_times = [h.Vector() for i in range(len(neurons))]
    MN_spike_detector = []
    for i in range(len(neurons)):
        sp_detector = h.NetCon(neurons[i].soma(0.5)._ref_v, None, sec=neurons[i].soma)
        sp_detector.threshold = -5
        MN_spike_detector.append(sp_detector)
        MN_spike_detector[i].record(MN_spike_times[i])
    return MN_spike_times

def create_exponential_synapses(source,target,W,tau,delay=0,inhibitory=False):
    syn_list=[]
    nc_list=[]

    for itarget in range(len(target)):
        syn_list.append([])
        nc_list.append([])

        for isource in range(len(source)):
            if inhibitory: 
                syn_ = h.Exp2Syn(target[itarget].soma(0.5))
                syn_.tau1 = 1.5
                syn_.tau2 = 2
                syn_.e = -75
            else:
                syn_ = h.ExpSyn(target[itarget].soma(0.5))
                syn_.tau = tau
                
            nc = h.NetCon(source[isource], syn_)
            nc.weight[0] = W[isource,itarget]
            nc_list[-1].append(nc)
            syn_list[-1].append(syn_)
            if type(delay) == np.ndarray: 
                nc.delay = delay[isource,itarget]
            else: 
                nc.delay = delay

    return syn_list,nc_list


def create_inhomongenous_poisson_process(max_fr, tStop, integration_step, offset_x, frequency, offset_y, limit_fr=np.inf): 
    """ Simulate an inhomongenous Poisson process. TODO: full documentation here"""

    T = np.arange(0, tStop, integration_step, dtype=float)
    Y = np.maximum(max_fr * np.sin(2 * np.pi * frequency * T / 1000 - offset_x) + offset_y,0.001)

    Y = np.minimum(Y, limit_fr)
    max_lambda = np.max(Y)
    samples = tStop/integration_step

    total_spikes = np.random.poisson(lam=max_lambda*tStop/1000)
    # create homogenous process
    homogeneous_spikes = np.sort([np.random.uniform(0, samples) for _ in range(total_spikes)])
    # thin process based on inhomogenous rate
    inhomogeneous_spikes = [homogeneous_spikes[i]*integration_step for i in range(total_spikes) if np.random.uniform(0, 1) <= Y[int(homogeneous_spikes[i])]/max_lambda]
    
    return [int(spike) for spike in inhomogeneous_spikes]

def create_inhomogeneous_input_neurons(N, max_rate, tStop=6000, offset_x=0, frequency=0.45, offset_y=0, limit_fr=np.inf): 
    # NOTE freq is in Hz (cycles per s)
    
    input_neurons = []
    
    for _ in range(N):
        cell = h.VecStim()
        firings = create_inhomongenous_poisson_process(max_rate, tStop=tStop, integration_step=1, offset_x=offset_x, frequency=frequency, offset_y=offset_y, limit_fr=limit_fr)
        vec = h.Vector(firings)
        cell.play(vec)
        input_neurons.append(cell)
    return input_neurons

def create_random_inhomgeneous_pattern(max_fr_rate, max_osc_rate, tStop):
    osc_rates_chosen = []
    max_fr_chosen = []
    num_reps_chosen = []

    possible_osc_rates = np.linspace(0.5, max_osc_rate, 50)
    possible_num_reps = [x for x in range(1,5)]
    possible_max_frs = np.linspace(10, max_fr_rate, 10)
    
    t_tracker = 0
    while t_tracker < tStop: 
        osc_rate = choice(possible_osc_rates)
        osc_rates_chosen.append(osc_rate)
        max_fr_chosen.append(choice(possible_max_frs))
        num_reps = choice(possible_num_reps)
        num_reps_chosen.append(num_reps)
        
        t_curr = num_reps/osc_rate*1000
        t_tracker += t_curr 

    return max_fr_chosen, osc_rates_chosen, num_reps_chosen

def create_random_inhomogeneous_firing(firing_pattern_max_fr, 
                                       firing_pattern_osc_rates, 
                                       firing_pattern_num_reps, 
                                       tStop): 
    t_tracker = 0
    spikes = []

    for i in range(len(firing_pattern_max_fr)): 
        num_reps = firing_pattern_num_reps[i]
        osc_rate = firing_pattern_osc_rates[i]
        max_fr = firing_pattern_max_fr[i]

        t_curr = num_reps/osc_rate*1000
        T = np.arange(0, t_curr, dtype=float)
        Y = np.maximum(max_fr * np.sin(2 * np.pi * osc_rate * T / 1000),0.001)
        Y = np.minimum(Y, max_fr)
        max_lambda = np.max(Y)

        total_spikes = np.random.poisson(lam=max_lambda*t_curr/1000)
        # create homogenous process
        homogeneous_spikes = np.sort([np.random.uniform(0, t_curr) for _ in range(total_spikes)])
        # thin process based on inhomogenous rate
        [spikes.append(homogeneous_spikes[i]+t_tracker) for i in range(total_spikes) if np.random.uniform(0, 1) <= Y[int(homogeneous_spikes[i])]/max_lambda]
        
        t_tracker += t_curr 

    return [int(spike) for spike in spikes if int(spike) < tStop]

def create_random_inhomogeneous_input_neurons(N, max_fr_rate, max_osc_rate, tStop=6000): 
    input_neurons = []
    max_fr_chosen, osc_rates_chosen, num_reps_chosen = create_random_inhomgeneous_pattern(max_fr_rate, max_osc_rate, tStop=tStop)
    
    for _ in range(N):
        cell = h.VecStim()
        firings = create_random_inhomogeneous_firing(max_fr_chosen, osc_rates_chosen, num_reps_chosen, tStop)
        vec = h.Vector(firings)
        cell.play(vec)
        input_neurons.append(cell)

    return input_neurons

def create_time_series_firing(time_series, tStop, integration_step=0.025): 
    """ Simulate an inhomongenous Poisson process based on a time series input of lambda values."""

    max_lambda = np.max(time_series)
    samples = tStop/integration_step

    total_spikes = np.random.poisson(lam=max_lambda*tStop/1000)
    # create homogenous process
    homogeneous_spikes = np.sort([np.random.uniform(0, samples) for _ in range(total_spikes)])
    # thin process based on inhomogenous rate
    inhomogeneous_spikes = [homogeneous_spikes[i]*integration_step for i in range(total_spikes) if np.random.uniform(0, 1) <= time_series[int(homogeneous_spikes[i])]/max_lambda]
    
    return [int(spike) for spike in inhomogeneous_spikes]


def create_time_series_inhomogeneous_input_neurons(N, lambda_time_series, tStop=6000): 
    """
    Input lambda_time_series as a time series of the values of lambda. 
    """
    input_neurons = []

    for _ in range(N):
        cell = h.VecStim()
        firings = create_time_series_firing(lambda_time_series, tStop)
        vec = h.Vector(firings)
        cell.play(vec)
        input_neurons.append(cell)

    return input_neurons


def create_depressing_scs_train(frequency, tau_reuptake, simulation_duration, dt=0.025, prob_syn_release=0.7, n_vesicle=5, stabilize_time=0): 
    # set up variables 
    t_vec_len = int(simulation_duration/dt)
    n_all = n_vesicle*np.ones(t_vec_len) # docked vesicles in each synapse
    scs_pulses = np.array([1 if np.mod(x, int(1/frequency*1000/dt)) == 0 else 0 for x in range(t_vec_len)]) # pulses from SCS
    release = []

    # run simulation of synaptic failure, predict spike train
    for t in range(t_vec_len - 1): 
        if (scs_pulses[t] == 1): # if SCS pulse succeeds  
            if t < stabilize_time/dt: 
                release.append(t*dt)
            else:
                p = 1-(1-prob_syn_release)**n_all[t]
                if np.random.rand() <= p: # if synaptic success 
                    redock_time = np.random.exponential(tau_reuptake)
                    redock_time = int(np.round(np.min([t_vec_len-t, redock_time])))
                    n_all[t:] = np.max([0, n_all[t]-1])
                    n_all[t+redock_time:] += 1 
                    release.append(t*dt)

    return release

def create_depressing_scs_neurons(N, frequency, tau_reuptake, simulation_duration, dt=0.025, stabilize_time=0): 
    scs_neurons = []
    for _ in range(N): 
        cell = h.VecStim()
        firings = create_depressing_scs_train(frequency, tau_reuptake, simulation_duration, dt, stabilize_time=stabilize_time)
        vec = h.Vector(firings)
        cell.play(vec)
        scs_neurons.append(cell)

    return scs_neurons


if __name__ == '__main__': 
    
    tau_scs = 3000
    prob_syn_release = 0.7
    hzs = [10, 40, 125]
    num_mns = 100

    for hz in hzs: 
        release_all = []
        for mn in range(num_mns): 
            release_all.append(create_depressing_scs_train(hz, tau_reuptake=tau_scs, simulation_duration=5000, prob_syn_release=prob_syn_release, n_vesicle=5))
        
        stim_timings = [x for x in range(0, 5000, int(1/hz*1000))]
        release_perc = [sum(release.count(target)/num_mns for release in release_all) for target in stim_timings]
        plt.plot(stim_timings, release_perc, label=hz)
        
        print((int(0.065*hz)+1))
        print(sum(release_perc[-(int(0.065*hz)+1):]))
        
    plt.legend()
    plt.show()