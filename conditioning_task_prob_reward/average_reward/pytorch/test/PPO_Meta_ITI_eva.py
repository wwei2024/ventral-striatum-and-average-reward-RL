import sys
import os
sys.path.append(os.path.abspath('..'))
import params
from ppo_pav_lstm import *
import threading
import multiprocessing
import numpy as np
import torch
import argparse

eva_num = params.eva_num

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed","-s", type=int, default=1,
                    help="seed for random number generator")   
    args = parser.parse_args()
    seed = args.seed
    np_seed = 2*seed
    pytorch_seed = 2*seed +1
    
    torch.manual_seed(pytorch_seed)
    
    with_units_data = False
    model_path  = os.path.join(os.path.abspath(".."),"model_meta")    
    train_nums  = [i*save_per_steps for i in range(1,round((simulation_steps+pretrain_steps)/save_per_steps)+1)]
    evainfofile = "eva_info.txt"

    for train_num in train_nums:
        print(train_num)
        filename = 'RLmodel'+'_data_'+str(train_num)+'_eva_'+str(eva_num)+'trials_'+\
                   'seed_'+str(seed)+'.npz'
        if os.path.exists(filename) and os.path.getsize(filename) > 20000000:
            continue
        with open(evainfofile, 'a') as f:
            if train_num <= train_nums[0]:
                f.write("seed: %d\n" % seed)
            f.write("eva Model: %d\n" % train_num)

        model_filename = f'model-{train_num}.pt'
        PATH = os.path.join(model_path, model_filename)
        
        pretrain=True
        env  = cue_input(np_seed,pretrain)
        agent = Agent()
        agent.load_state_dict(torch.load(PATH)['model_state_dict'])

        
        episode_rewards = [] 
        episode_states  = []
        episode_trials  = []
        trial_lengths   = []
        trial_type      = []
        events_ITI_Respond=[]

        block_count=0
        done_eva=False
        
        next_obs = torch.Tensor(env.reset())
        next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
        )
        
        while done_eva==False: 
            ITI_time      = env.ITI_time
            time_step_cue = env.time_step_cue
            trial_type.append(env.cue_type)
            trial_len  = env.trial_len 
            trial_lengths.append(trial_len)
            trial_buffer = []
            trial_state  = []
            done_trial = False
            
            while not done_trial:
                                
                with torch.no_grad():
                    action,logprob,_,value,_,next_lstm_state = \
                    agent.get_action_and_value(next_obs,next_lstm_state)
                       
                a     = action.item()
                value = value.item()
                pi_a  = logprob.item()

                obs, reward, done_trial, info = env.step(a)

                trial_buffer_temp=[a,reward]
                trial_buffer_temp.extend(next_obs.numpy())
                trial_buffer_temp.extend([value,pi_a])
                trial_buffer.append(trial_buffer_temp)
                trial_state.append(next_lstm_state)

                next_obs = torch.Tensor(obs)
                                               
            time_step_food = env.time_step_food
            trial_reward   = env.trial_reward

            events_ITI_Respond.append([ITI_time,time_step_cue,time_step_food,trial_len])      
            episode_rewards.append(trial_reward)  
            episode_states.append(trial_state)
            episode_trials.append(trial_buffer)
            block_count+=1
            if block_count>=eva_num: 
                done_eva=True
        data_model = {'episode_trials': episode_trials,
                      'episode_rewards':episode_rewards,
                      'events_ITI_Respond': events_ITI_Respond,
                      'trial_lengths': trial_lengths,
                      'trial_type': trial_type
                      }
        np.savez_compressed(filename, dat=data_model)
        
        if with_units_data:
            data_model_units = {'episode_states': episode_states}
            filename1 = 'RLmodel'+'_data_units_'+str(train_num)+'_eva_'+str(eva_num)+'trials_'+\
                        'seed_'+str(seed)+'.npz'
            np.savez_compressed(filename1, dat=data_model_units)
    

