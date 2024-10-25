import math

pretrain_steps   = 500
simulation_steps = 10000
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
holding_time_3_sec     = [0.6, 3.0, 12.0, 0.0] # sec
cue_duration_aud_3_sec = [0.6, 3.0, 12.0, 0.0] # sec
cue_duration_rew_sec = 0.1
reward_delay_sec     = 0.0
trial_shift_sec      = 3.0

dt = 0.1 # sec
ITI_min = round(ITI_min_sec/dt)
ITI_max = round(ITI_max_sec/dt)
holding_time_3     = [round(item/dt) for item in holding_time_3_sec]
cue_duration_aud_3 = [round(item/dt) for item in cue_duration_aud_3_sec]
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
