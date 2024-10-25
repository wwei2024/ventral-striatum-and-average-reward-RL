import sys
import os
sys.path.append(os.path.abspath('..'))
import params
from ppo_bandit_lstm import *
import numpy as np
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import argparse

if __name__ == '__main__':

    args = parse_args()
    seed = args.seed

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    d_state  = 0 # states  are continuous
    
    # env setup
    pretrain=True
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i,pretrain) for i in range(1)]
    )
    agent = Agent(envs).to(device)

    with_units_data = True  #False #
    model_path = os.path.join(os.path.abspath(".."),"model_meta")    
    train_nums = [175000]
    eva_num    = 200 # number of blocks. 
    beta_e=args.ent_coef
    evainfofile = "eva_info.txt"

    for train_num in train_nums:
        filename  = 'RLmodel'+'_data_'+str(train_num)+'_eva_'+str(eva_num)+'blocks_'+\
                    'dt100_betae'+str(beta_e)+'_seed_'+str(seed)+'.npz'
        filename1 = 'RLmodel'+'_data_units_'+str(train_num)+'_eva_'+str(eva_num)+'blocks_'+\
                    'dt100_betae'+str(beta_e)+'_seed_'+str(seed)+'.npz'
            
        with open(evainfofile, 'a') as f:
            if train_num <= train_nums[0]:
                f.write("seed: %d\n" % seed)
            f.write("eva Model: %d\n" % train_num)

        PATH = model_path+'/model-'+str(train_num)+'.pt'

        agent.load_state_dict(torch.load(PATH)['model_state_dict'])
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(1).to(device)
        next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        )
            
        episode_rewards = [] 
        episode_states  = []
        episode_values  = []
        episode_actions = []
        episode_side_in_steps = []
        episode_trials  = []
        episode_blocks  = []
        episode_bandits = []    
        trial_lengths    = []
        failed_trials   = []
        cumregret=0            
        events_ITI_CNI_Go_Cue_Respond=[]
        block_count=0
        done_block=False
                    
        while done_block==False: 
            bandit_backup      = envs.get_attr(name='bandit')[0]
            blocklength_backup = envs.get_attr(name='blocklength')[0]
            ITI_time = envs.get_attr(name='ITI_time')[0]
            
            done_trial=False
            step = 0
            trial_buffer = []
            trial_state  = []
            while not done_trial:
                                
                with torch.no_grad():
                    action, logprob, _, value, _,next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done*d_state)
                obs, reward, done, info = envs.step(action.cpu().numpy())
                
                a = action.item()
                logprob = logprob.item()
                value   = value.item()
                trial_buffer_temp=[a,reward[0]]
                trial_buffer_temp.extend(next_obs.cpu().numpy()[0])
                trial_buffer_temp.extend([value,logprob])
                trial_buffer.append(trial_buffer_temp)
                trial_state.append(torch.cat(next_lstm_state, dim=-1).view(-1).numpy())

                next_obs, next_done = torch.Tensor(obs).to(device), torch.Tensor(done).to(device)               
                
                if done:
                    done_trial = True

                step+=1
            trial_length = step
            failed_trial = info['failed_trial'][0]
            choice_trial = info['choice_trial'][0]
            if not failed_trial:
                time_step_CNI     = envs.get_attr(name='time_step_CNI')[0]
                time_step_go_cue  = envs.get_attr(name='time_step_go_cue')[0]
                time_step_side_in = envs.get_attr(name='time_step_side_in')[0]
                try:
                    time_step_food = envs.get_attr(name='time_step_food')[0]
                except AttributeError:
                    time_step_food = -1
                
                if envs.get_attr(name='blockstep')[0]==0: 
                    block_count+=1
                    if block_count%50==0: 
                        print(block_count)
                    episode_bandits.append(bandit_backup)
                    episode_blocks.append(blocklength_backup)
            else:
                time_step_CNI = np.nan
                time_step_go_cue = np.nan
                time_step_side_in = np.nan
                time_step_food    = np.nan
            
            trial_length = step   
            episode_rewards.append(reward[0]) 
            episode_trials.append(trial_buffer) 
            trial_lengths.append(trial_length)
            failed_trials.append(failed_trial)
            events_ITI_CNI_Go_Cue_Respond.append([ITI_time,time_step_CNI,time_step_go_cue,time_step_side_in,time_step_food,trial_length])
            episode_actions.append(choice_trial)  
            episode_states.append(trial_state)
            
            
            if block_count>=eva_num: 
                done_block=True
        data_model = {'failed_trials':failed_trials,'trial_lengths':trial_lengths,\
                 'episode_trials': episode_trials,'episode_actions':episode_actions,\
                 'episode_rewards':episode_rewards,\
                 'events_ITI_CNI_Go_Cue_Respond': events_ITI_CNI_Go_Cue_Respond,\
                 'episode_bandits':episode_bandits,'episode_blocks':episode_blocks}
        np.savez_compressed(filename, dat=data_model)
        
        if with_units_data:
            data_model_units = {'episode_states': episode_states}
            np.savez_compressed(filename1, dat=data_model_units)
    

