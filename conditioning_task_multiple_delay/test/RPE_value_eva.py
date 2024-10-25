import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'seed'))
import params
import numpy as np

if __name__ == '__main__':
    dt = params.dt
    feature_dim      = params.feature_dim
    background_dim   = params.background_dim
    poking_cost      = params.poking_cost
    trial_shift      = params.trial_shift
    eva_num          = params.eva_num
    save_per_steps   = params.save_per_steps
    simulation_steps = params.simulation_steps
    pretrain_steps   = params.pretrain_steps
    areas_num   = 1 
    areas       = ['VS'] 
    beta_e      = 0.001

    train_nums  = [i*save_per_steps for i in range(1,round((simulation_steps+pretrain_steps)/save_per_steps)+1)]
    feature_end = 2+feature_dim+background_dim
    seeds = [i for i in range(1,11)]
    cwd   = os.path.abspath(".")
    cwd_u = os.path.abspath("..")
    cue_type_idx = 10  
    shift        = round(2.0/dt) # the time points to show before cue onset
    len_traj     = round(30.0/dt)

    for seed_i in range(len(seeds)):
        seed = seeds[seed_i]
        print(f"seed: {seed}")
        for train_num_idx in range(len(train_nums)):
            train_num = train_nums[train_num_idx] 
            print(f"train_num: {train_num}")
            file_RPE = 'RPE_value_poking_seed_'+str(seed)+'_train_'+str(train_num)+'.npz'
            filepath = os.path.join(cwd_u,'seed'+str(seed),'test')
            filename = 'RLmodel'+'_data_'+str(train_num)+'_eva_'+str(eva_num)+'trials_'+\
                       'seed_'+str(seed)+'.npz'
            file1 = os.path.join(filepath,filename)
            if os.path.exists(file_RPE):
                print(f"  RPE data already exist for seed: {seed}, train: {train_num}")
                continue
            if not os.path.exists(file1) or os.path.getsize(file1) < 20000000:
                print(f"  data not exist for seed: {seed}, train: {train_num}")
                continue
            else:    
                RPE_seed_train_traj    = np.zeros([len(seeds),areas_num, cue_type_idx, len_traj])
                RPE_seed_train_traj_z  = np.zeros([len(seeds),areas_num, cue_type_idx, len_traj])
                RPE_seed_train_traj_zn = np.zeros([len(seeds),areas_num, cue_type_idx, len_traj])
                value_seed_train_traj  = np.zeros([len(seeds),areas_num, cue_type_idx, len_traj])
                poking_seed_train_traj = np.zeros([len(seeds),cue_type_idx,len_traj])
                        
                dat0 = np.load(file1, allow_pickle=True)['dat'].item()
                episode_trials  = dat0['episode_trials']
                episode_rewards = dat0['episode_rewards']
                events_ITI_Respond = dat0['events_ITI_Respond']
                trial_lengths  = dat0['trial_lengths']
                trial_type     = dat0['trial_type']
                del dat0

                episode_trials1=list(episode_trials)
                events_ITI_Respond1=list(events_ITI_Respond)
                trial_lengths1=list(trial_lengths)
                trial_type1=list(trial_type)
                
                max_len = max(trial_lengths1)+1
                len1    = len(episode_trials1[0][0])
                rollout = np.array([trial+[[0]*len1 for _ in range(int(max_len - len(trial)))]\
                                    for trial in episode_trials1],dtype=np.float32)
                actions = rollout[:,:,0]
                rewards = rollout[:,:,1]
                bootstrap_value=0
                values_VS   = rollout[:,:,feature_end+1]
                values_areas = [values_VS] #[values_DLS, values_DMS, values_VS]
                ITI = []    
                for k in range(len(events_ITI_Respond1)):
                    ITI.append(events_ITI_Respond1[k][0])
                holding_index_c1=[]
                holding_index_c4=[]
                holding_index_c16=[]
                holding_index_cfree  =[] 
                holding_index_c1_rew =[]
                holding_index_c4_rew =[]
                holding_index_c16_rew=[]
                holding_index_c1_nor = [] 
                holding_index_c4_nor = []
                holding_index_c16_nor= []              

                for k in range(len(trial_lengths1)):
                    if trial_type1[k] == '0': #episode_rewards1[k]==1:
                        holding_index_c1.append(k)
                        if any(rewards[k]>=1): #episode_rewards1[k]==1:
                            holding_index_c1_rew.append(k)
                        else:
                            holding_index_c1_nor.append(k)
                        
                    elif trial_type1[k] == '1':            
                        holding_index_c4.append(k)
                        if any(rewards[k]>=1): #episode_rewards1[k]==1:
                            holding_index_c4_rew.append(k)
                        else:
                            holding_index_c4_nor.append(k)
                        
                    elif trial_type1[k] == '2':
                        holding_index_c16.append(k) 
                        if any(rewards[k]>=1): #episode_rewards1[k]==1:
                            holding_index_c16_rew.append(k)
                        else:
                            holding_index_c16_nor.append(k)
                    else:
                        holding_index_cfree.append(k)
                holding_index = [holding_index_c1,holding_index_c4, 
                                 holding_index_c16,holding_index_cfree,
                                 holding_index_c1_rew,holding_index_c4_rew, 
                                 holding_index_c16_rew,
                                 holding_index_c1_nor,holding_index_c4_nor, 
                                 holding_index_c16_nor]

                # for poking
                poking_aud_cue=np.zeros((len(rewards),len_traj))
                for k in range(len(rewards)):
                    aud_cue_time = events_ITI_Respond1[k][1] #events_ITI_Respond1[k][0]-trial_shift #
                    for m in range(trial_lengths1[k]-aud_cue_time+shift): #check this shift 1                        
                        poking_aud_cue[k][m]=actions[k][m+aud_cue_time-shift]
                for idx_i in range(len(holding_index)):
                    holding_index_i = holding_index[idx_i]
                    traj_poking_i = np.mean(poking_aud_cue[holding_index_i],axis=0)             
                    poking_seed_train_traj[seed_i,idx_i] = traj_poking_i
                
                for area_idx in range(1): #range(len(gamma_areas)): 
                    advantages = np.zeros_like(rewards)
                    area   = areas[area_idx]
                    values = values_areas[area_idx]
                    for k in range(len(episode_trials1)): 
                        valuesk=values[k].tolist() 
                        valuesk.append(bootstrap_value)
                        advantages[k] = rewards[k] + np.array(valuesk)[1:] - np.array(valuesk)[:-1] 
                    
                    advantages_mean = advantages[np.nonzero(advantages)].mean()
                    advantages_std  = advantages[np.nonzero(advantages)].std()
                    advantages_z    = (advantages - advantages_mean) / (advantages_std + 1e-8)
                    
                    advantages_aud_cue   = np.zeros((len(rewards),len_traj))
                    advantages_aud_cue_z = np.zeros((len(rewards),len_traj))
                    value_aud_cue=np.zeros((len(rewards),len_traj))
                    for k in range(len(rewards)):
                        aud_cue_time = events_ITI_Respond1[k][1] 
                        for m in range(trial_lengths1[k]-aud_cue_time+shift): 
                            advantages_aud_cue[k][m]    = advantages[k][m+aud_cue_time-shift]  
                            advantages_aud_cue_z[k][m]  = advantages_z[k][m+aud_cue_time-shift]     
                        for m in range(trial_lengths1[k]-aud_cue_time+shift): 
                            value_aud_cue[k][m]=values[k][m+aud_cue_time-shift]            
                    norm = np.max(np.mean(advantages_aud_cue_z[holding_index_cfree],axis=0))
                    for idx_i in range(len(holding_index)):
                        holding_index_i = holding_index[idx_i]
                        traj_RPE_i    = np.mean(advantages_aud_cue[holding_index_i],axis=0)
                        traj_RPE_i_z  = np.mean(advantages_aud_cue_z[holding_index_i],axis=0)
                        traj_value_i  = np.mean(value_aud_cue[holding_index_i],axis=0)
                        RPE_seed_train_traj[seed_i,area_idx,idx_i]    = traj_RPE_i
                        RPE_seed_train_traj_z[seed_i,area_idx,idx_i]  = traj_RPE_i_z
                        RPE_seed_train_traj_zn[seed_i,area_idx,idx_i] = traj_RPE_i_z/norm
                        value_seed_train_traj[seed_i,area_idx,idx_i]  = traj_value_i

                data_RPE = {'RPE_seed_train_traj':RPE_seed_train_traj,
                            'RPE_seed_train_traj_z':RPE_seed_train_traj_z,
                            'RPE_seed_train_traj_zn':RPE_seed_train_traj_zn,
                            'value_seed_train_traj':value_seed_train_traj,
                            'poking_seed_train_traj':poking_seed_train_traj, 
                            'train_nums':train_num}
                np.savez_compressed(file_RPE, dat=data_RPE)
