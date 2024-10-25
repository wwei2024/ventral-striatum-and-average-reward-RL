import sys
import os
sys.path.append(os.path.abspath('..'))
import params
from PPO_Meta_ITI import *
import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import argparse

eva_num = params.eva_num

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed","-s", type=int, default=1,
                    help="seed for random number generator")   
    args = parser.parse_args()
    seed = args.seed
    np_seed = 2*seed
    tf_seed = 2*seed +1

    with_units_data = True #False
    model_path   = os.path.join(os.path.abspath(".."),"model_meta")    
    train_nums   = [i*save_per_steps for i in range(1,round((simulation_steps+pretrain_steps)/save_per_steps)+1)]
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
        path = os.path.join(model_path,"checkpoint")
        line1 = "model_checkpoint_path: \"model-" + str(train_num) + ".cptk\"" + "\n"
        line2 = "all_model_checkpoint_paths: \"model-" + str(train_num) +".cptk\""
        with open(path, "w") as f:
            f.write(line1)
            f.write(line2)
  
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(tf_seed)
                   
        with tf.device("/cpu:0"): 
            master_network = AC_Network(np_seed,beta_e,a_size,'global',None) # Generate global network
            saver = tf.compat.v1.train.Saver(max_to_keep=None)

        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)  
            master_AC=master_network
            rnn_state_init_VS  = master_AC.state_init_VS
            
            episode_rewards     = [] 
            episode_states_VS   = []
            episode_trials  = []
            trial_lengths   = []
            trial_type      = []
            events_ITI_Respond=[]
            block_count=0
            done=False
            pretrain = True
            env=cue_input(np_seed,pretrain)
            obs = env.reset()
            while done==False: 
                ITI_time     = env.ITI_time
                time_step_cue = env.time_step_cue
                trial_type.append(env.cue_type)
                obs,time_step_food,trial_length,trial_reward,trial_buffer,\
                trial_state_VS,rnn_state_VS,value_VS\
                = master_AC.perform_trial(sess,obs,env,rnn_state_init_VS)                
                rnn_state_init_VS  = rnn_state_VS                
                    
                trial_lengths.append(trial_length)
                events_ITI_Respond.append([ITI_time,time_step_cue,time_step_food,trial_length])
                episode_rewards.append(trial_reward)        
                episode_states_VS.append(trial_state_VS)
                episode_trials.append(trial_buffer)

                block_count+=1
                if block_count>=eva_num: 
                    done=True
        data_model = {'episode_trials': episode_trials,
                      'episode_rewards':episode_rewards,
                      'events_ITI_Respond': events_ITI_Respond,
                      'trial_lengths': trial_lengths,
                      'trial_type': trial_type}
        np.savez_compressed(filename, dat=data_model)
        
        if with_units_data:
            data_model_units = {'episode_states_VS': episode_states_VS}
            filename1 = 'RLmodel'+'_data_units_'+str(train_num)+'_eva_'+str(eva_num)+'trials_'+\
                       'seed_'+str(seed)+'.npz'
            np.savez_compressed(filename1, dat=data_model_units)
    
