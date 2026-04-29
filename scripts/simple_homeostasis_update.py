import numpy as np
from neuron import h
import matplotlib.pyplot as plt
import sys

sys.path.append('../HomeostasisMotoneuronModel')
import tools.neuron_functions as nf
import tools.plotting_tools as pt
from cells import MotoneuronNoDendrites

h.load_file("stdrun.hoc")
np.random.seed(0)

mn = MotoneuronNoDendrites()

T_total = 10000
dt = 0.025

num_supraspinal = 30
rate_supraspinal_healthy = 200
rate_supraspinal_lesion = 2
inhomo_process = int(T_total/(4*dt))*[rate_supraspinal_healthy] + int(T_total*3/(4*dt))*[rate_supraspinal_lesion]

supra_tau = 2
synaptic_weight_supra = 0.00015
shape = 1.2

# descending input from the brain
supraspinal_neurons = nf.create_time_series_inhomogeneous_input_neurons(num_supraspinal, inhomo_process, tStop=T_total)
supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)
W_supraspinal = np.random.gamma(shape, scale=synaptic_weight_supra/shape, size=[num_supraspinal, 1])
syn_supraspinal, nc_supraspinal = nf.create_exponential_synapses(supraspinal_neurons, [mn], W_supraspinal, supra_tau)

# recording
t = h.Vector().record(h._ref_t)
v = h.Vector().record(mn.soma(0.5)._ref_v)
cai = h.Vector().record(mn.soma(0.5).motoneuron._ref_cai)
gnabar = h.Vector().record(mn.soma(0.5).motoneuron._ref_gnabar)
gkrect = h.Vector().record(mn.soma(0.5).motoneuron._ref_gkrect)
gcaN = h.Vector().record(mn.soma(0.5).motoneuron._ref_gcaN)

# targets
target_ca = 0.00041 # based on the healthy avg Ca2+

# learning rates
alpha_na = 1e0
alpha_k = 1e0
alpha_ca = 1e-1

dt_homeo = 50

t_current = 0

h.run(1)

while t_current < T_total:
    h.continuerun(t_current + dt_homeo)

    # activity signal
    ca_val = cai.to_python()[int(t_current/dt):int((t_current+dt_homeo)/dt)]
    error = target_ca - np.mean(ca_val)
    seg = mn.soma(0.5).motoneuron

    # update conductances
    seg.gnabar = max(0, seg.gnabar + alpha_na * error)
    seg.gkrect = max(0, seg.gkrect - alpha_k * error)
    seg.gcaN   = max(0, seg.gcaN + alpha_ca * error)

    t_current += dt_homeo


# plot results of simulation
    
# plot supraspinal firing and membrane voltage/Ca2+ current
plt.subplots(3, 1, sharex='col')

ax = plt.subplot(3, 1, 1)
supra_spikes = [np.array(supraspinal_spike_times[i])  if len(supraspinal_spike_times[i]) > 0 else [] for i in range(num_supraspinal)]
pt.plot_raster_plot(ax, supraspinal_spike_times, ylabel='Supraspinal\n inputs')

ax = plt.subplot(3, 1, 2)
pt.plot_time_series_data(ax, t, v, ylabel='Voltage (mV)')

ax = plt.subplot(3, 1, 3)
pt.plot_time_series_data(ax, t, cai*1000, ylabel='Calcium current\n (uA/cm^2)')
plt.xlabel('Time (ms)')

# plot supraspinal firing and membrane voltage/Ca2+ current
plt.subplots(4, 1, sharex='col')

ax = plt.subplot(4, 1, 1)
pt.plot_time_series_data(ax, t, gnabar, ylabel='Sodium (Na+)\n conductance')

ax = plt.subplot(4, 1, 2)
pt.plot_time_series_data(ax, t, gkrect, ylabel='Potassium (K+)\n conductance')

ax = plt.subplot(4, 1, 3)
pt.plot_time_series_data(ax, t, gcaN, ylabel='Calcium (Ca2+)\n conductance')

ax = plt.subplot(4, 1, 4)
pt.plot_time_series_data(ax, t, v, ylabel='Voltage (mV)')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.show()