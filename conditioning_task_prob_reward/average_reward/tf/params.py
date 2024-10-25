import math

pretrain_steps   = 500
simulation_steps = 1500
save_per_steps   = 100
eva_num          = 1000 

click_dim       = 1
click_strength  = 5.0

feature_dim    = 20 # three dim for overlapping feature
background_dim = 3 # background input
n_step         = 20

lr         = 0.0005
beta_e     = 0.001
batch_size = 1
epsilon    = 0.1

time_horizon_sec = 500 #sec
ITI_min_sec = 15 # sec
ITI_max_sec = 30 # sec
holding_time_sec = 3.1 # sec. 2.6 cue + 0.5 holding
cue_duration_aud_sec = 2.6
cue_duration_rew_sec = 0.1
reward_delay_sec     = 0.0
trial_shift_sec      = 3.0

dt = 0.1 #sec
ITI_min = round(ITI_min_sec/dt)
ITI_max = round(ITI_max_sec/dt)
holding_time     = round(holding_time_sec/dt)
cue_duration_aud = round(cue_duration_aud_sec/dt)
cue_duration_rew = round(cue_duration_rew_sec/dt)
reward_delay     = round(reward_delay_sec/dt)
trial_shift      = round(trial_shift_sec/dt)
time_horizon     = round(time_horizon_sec/dt) 

lambda_gae = 0.98
beta_e     = 0.001
beta_v     = 0.8
epsilon    = 0.1
batch_size = 1
action_names = ['no_poke', 'poke']
a_size = 2 

poking_cost = -0.006*dt/0.1
