#The FeedForward model

from foundations.helpers import scale_input_theory
from brian2 import *
from models.currentmodel import Barrel_PC, Barrel_IN
from foundations.helpers import scale_to_freq
from foundations.make_dynamic_experiments import make_dynamic_experiments
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#parameters setting
baseline = 0
theta = 0
factor_ron_roff = 2
tau_PC = 250
ron_PC = 1./(tau_PC*(1+factor_ron_roff))
roff_PC = factor_ron_roff*ron_PC
mean_firing_rate_PC = (0.1)/1000
duration_PC = 100000
tau_IN = 50
ron_IN = 1./(tau_IN*(1+factor_ron_roff))
roff_IN = factor_ron_roff*ron_IN
mean_firing_rate_IN = (0.5)/1000
duration_IN = 20000
sampling_rate = 5
dt = 1/sampling_rate
qon_qoff_type = 'balanced'
Er_exc, Er_inh = (0, -75)
target_PC = 1.4233
target_IN = 6.6397
on_off_ratio = 1.5
scale_list = np.append([1], np.arange(2.5, 302.5, 2.5))
scales = {'CC_PC':19, 'DC_PC':30, 'CC_IN':17, 'DC_IN':6}
N_runs = 1
PC_i = 35
IN_i = 11

#getting the input

[input_theory, dynamic_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, tau_PC, factor_ron_roff, mean_firing_rate_PC, sampling_rate, duration_PC)
inj_input = scale_input_theory(input_theory, 'current', 0, 18, dt)

#Making the PC cells

PC = Barrel_PC(dt=dt)
IN = Barrel_IN(dt=dt)

PC.getNeurongroup()
IN.getNeurongroup()

syn = Synapses(IN, PC, on_pre='''v_post += v''')
syn.w = 0.3
syn.connect()


#IN = current_IN.getNeurongroup()
#PC = current_PC.getNeurongroup()




