qon_qoff_type = 'balanced'
baseline = 0
theta = 0
factor_ron_roff = 2
tau_PC = 200
ron_PC = 1./(tau_PC*(1+factor_ron_roff))
roff_PC = factor_ron_roff*ron_PC
mean_firing_rate = (0.1)/1000
duration = 100000
sampling_rate = 5
scales = {'CC_PC':19, 'DC_PC':30, 'CC_IN':17, 'DC_IN':6}
dt = 1/sampling_rate
on_off_ratio = 1.5
target_PC = 1.4233

