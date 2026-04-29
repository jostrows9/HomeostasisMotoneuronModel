import numpy as np
from neuron import h
import matplotlib.pyplot as plt
import sys

sys.path.append('../HomeostasisMotoneuronModel')
import tools.plotting_tools as pt
from cells import MotoneuronNoDendrites


def measure_excitability(params, plot: bool = False): 
    """
    Given a dictionary of MN conductance parameters, return the lowest 
    current amplitude 
    """
    cur_amps = range(1, 10000, 100)
    divisor = 100000
    for cur_amp in cur_amps:
        np.random.seed(0)
        h.load_file("stdrun.hoc")

        mn = MotoneuronNoDendrites()
        t = h.Vector().record(h._ref_t)
        v = h.Vector().record(mn.soma(0.5)._ref_v)

        stim = h.IClamp(mn.soma(0.5))
        stim.delay = 500
        stim.dur = 500
        stim.amp = cur_amp/divisor

        seg = mn.soma(0.5).motoneuron
        seg.gnabar = params['gnabar']
        seg.gkrect = params['gkrect'] 
        seg.gcaN = params['gcaN']

        h.finitialize()
        h.tstop = 1500
        h.run()


        if any(x > 0 for x in v.to_python()[1000:]): 
            if plot:
                import pdb; pdb.set_trace() 
                plt.subplots(2, 1, sharex='col', figsize=(7,4))

                t = t.to_python()
                v = v.to_python()
                
                ax = plt.subplot(2, 1, 1)
                pt.plot_time_series_data(ax, [x/0.025/1000 for x in range(len(v[1000:]))], v[1000:], ylabel='Membrane voltage (mV)')

                ax = plt.subplot(2, 1, 2) 
                for amp in [x for x in range(1, 10000, 500)]:
                    if amp <= cur_amp: 
                        current_vis = [0]*760 + [amp/divisor]*850 + [0]*500
                        pt.plot_time_series_data(ax, [x for x in range(len(current_vis))], 
                                            current_vis, ylabel='Injected current (mA)')
                
                ax.set_ylim([0, max(cur_amps)/divisor])
                ax.set_xlabel('Time (ms)')
                plt.show()

            return cur_amp/divisor


if __name__ == '__main__':

    params = {'gnabar': 0.06170594516249951, 
              'gkrect': 0.29829405483750086, 
               'gcaN': 0.051170594516249856}
    
    params_healthy = {'gnabar': 0.05, 
                     'gkrect': 0.3, 
                     'gcaN': 0.05}
    
    print(f"Current needed to excite maladaptive cell: {measure_excitability(params=params, plot=True)}")
    print(f"Current needed to excite healthy cell: {measure_excitability(params=params_healthy, plot=True)}")