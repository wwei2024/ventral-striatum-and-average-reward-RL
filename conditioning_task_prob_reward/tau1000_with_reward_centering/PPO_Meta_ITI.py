import params
import threading
import multiprocessing
import os
import numpy as np
import tensorflow as tf
import argparse
tf.compat.v1.disable_eager_execution()

simulation_steps = params.simulation_steps
pretrain_steps   = params.pretrain_steps
save_per_steps   = params.save_per_steps
time_horizon     = params.time_horizon
dt               = params.dt

feature_dim      = params.feature_dim
background_dim   = params.background_dim
click_dim        = params.click_dim
click_strength   = params.click_strength
poking_cost      = params.poking_cost
gamma_VS         = params.gamma_VS
# gamma_DMS        = params.gamma_DMS
# gamma_DLS        = params.gamma_DLS
trial_shift      = params.trial_shift
ITI_min          = params.ITI_min
ITI_max          = params.ITI_max
reward_delay     = params.reward_delay
holding_time     = params.holding_time
cue_duration_aud = params.cue_duration_aud
cue_duration_rew = params.cue_duration_rew

lr               = params.lr
#n_step           = params.n_step
beta_e           = params.beta_e
beta_v           = params.beta_v
batch_size       = params.batch_size
epsilon          = params.epsilon
lambda_gae       = params.lambda_gae
action_names     = params.action_names
a_size           = params.a_size
load_model       = False
train            = True
model_path       = './model_meta'

lr_eta = 0.1
rm_vbias_coeff = 0.5

def update_target_graph(from_scope,to_scope):
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars   = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_x = np.zeros_like(x,dtype=float)
    running_add = 0
    for t in reversed(range(len(x))):
        running_add = running_add * gamma + x[t]
        discounted_x[t] = running_add        
    return discounted_x

# def discount_nstep(x, gamma, horizon, n_step):
#     disc_i = list(range(n_step))
#     disc = [np.power(gamma,i) for i in disc_i]
#     disc = np.array(disc)    
#     discounted_x = np.zeros(horizon,dtype=float) 

#     for t in range(horizon):
#         discounted_x[t] = np.sum(disc*x[t:t+n_step])
#     return discounted_x

#initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class cue_input(): 
    def __init__(self,np_seed,pretrain=False): 
        self.rng = np.random.default_rng(np_seed) 
        self.pretrain = pretrain 
        self.cue_types=['0', '1', '2', 'free_food']  
        self.cue_types_onehot={'0': [1.0, 0.0, 0.0]+[1.0]*(feature_dim-3)+[1.0]*background_dim,\
                               '1': [0.0, 1.0, 0.0]+[1.0]*(feature_dim-3)+[1.0]*background_dim,\
                               '2': [0.0, 0.0, 1.0]+[1.0]*(feature_dim-3)+[1.0]*background_dim,\
                       'free_food': [0.0, 0.0, 0.0]+[0.0]*(feature_dim-3)+[1.0]*background_dim}  
        
        self.action_names = action_names
        self.click_dim    = click_dim
        self.click_strength   = click_strength
        self.holding_time     = holding_time 
        self.cue_duration_aud = cue_duration_aud 
        self.cue_duration_rew = cue_duration_rew  
        self.reward_delay     = reward_delay # between click cue and reward delivery
        self.trial_shift      = trial_shift 
        self.time_step_food_stored = np.nan
        self.time_step_food        = np.nan
        self.trial_reward_stored   = 0.0
        self.trial_reward          = 0.0
        self.pre_a = action_names.index('no_poke')
        if self.pretrain:
            self.done_pretrain = True
        else:
            self.done_pretrain = False  
    def reset(self): 
        self.time_step = 0 
        self.pre_a = action_names.index('no_poke') # to compare with early tf code
        self.cue_type = self.rng.choice(self.cue_types)
        self.cue_type_onehot=self.cue_types_onehot[self.cue_type] 
        self.ITI_time  = self.rng.integers(low=ITI_min,high=ITI_max,endpoint=True)
        self.time_step_cue = self.ITI_time - self.trial_shift 
        self.reward_to_be  = self.pullArm()
        self.trial_len = self.time_step_cue + self.holding_time + self.trial_shift
        if self.reward_to_be>=1:
            self.time_step_reward_cue = self.time_step_cue + self.holding_time
        else:
            self.time_step_reward_cue = np.nan         
        self.food_picking   = False
        self.time_step_food_stored = self.time_step_food
        self.trial_reward_stored   = self.trial_reward # store the reward from finished trial
        self.time_step_food = np.nan
        self.trial_reward   = 0.0
        obs = self.cue_types_onehot['free_food'] +[0.0]*self.click_dim
        return np.array(obs).astype(np.float32)
    def pullArm(self):
        if self.cue_type == '0': 
            if self.rng.uniform() < 0.25:
                #return a positive reward.
                reward = 1.0
            else:
                #return no reward.
                reward = 0.0   
        elif self.cue_type == '1':
            if self.rng.uniform() < 0.75:
                #return a positive reward.
                reward = 1.0
            else:
                #return no reward.
                reward = 0.0 
        elif self.cue_type == '2':
            reward = 0.0
        else:
            reward = 1.0
        return reward
  
    def step(self,a):        
        a_name     = action_names[a]
        pre_a_name = action_names[self.pre_a]
        done_trial = False
                                     
        if a_name in ['no_poke']:
            reward = 0.0
        else:
            if pre_a_name in ['no_poke']:
                reward = poking_cost  
            else:
                reward = 0.1*poking_cost  

            if (self.reward_to_be >= 1.0) and \
               (self.food_picking == False) and \
               (self.time_step >= self.time_step_reward_cue + self.reward_delay):

                reward = self.reward_to_be
                self.time_step_food = self.time_step 
                self.food_picking = True
                self.trial_reward = reward
        
        if self.time_step >= self.trial_len-1:
            done_trial = True
        
        self.pre_a = a         
        info = {}        

        if not done_trial:
            self.time_step+=1 
            cues       = self.cue_types_onehot['free_food']
            reward_cue = [0.0] * self.click_dim
            
            if (self.done_pretrain) and (self.time_step >= self.time_step_cue) and \
               (self.time_step < self.time_step_cue + self.cue_duration_aud):
                cues = self.cue_type_onehot 
            if (self.reward_to_be >= 1) and (self.time_step >= self.time_step_reward_cue) and \
               (self.time_step < self.time_step_reward_cue + self.cue_duration_rew):
                reward_cue = [self.click_strength] * click_dim  
                  
            obs = cues + reward_cue
        else:
            obs = self.reset()    
        return np.array(obs).astype(np.float32), reward, done_trial, info

class AC_Network():
    def __init__(self,np_seed,beta_e,a_size,scope,trainer):
        self.rng = np.random.default_rng(np_seed)
        with tf.compat.v1.variable_scope(scope):
            self.num_LSTMunits_VS =32
            #Input and visual encoding layers
            self.batch_size = tf.compat.v1.placeholder(shape=[],dtype=tf.int32,name="batch_size")
            self.obs        = tf.compat.v1.placeholder(shape=[None,None,feature_dim+background_dim+1],dtype=tf.float32,name="cues")
            self.seq_length = tf.compat.v1.placeholder(tf.int32, [None],name="seq_length")
            inputs = self.obs
            
            #Recurrent network for temporal dependencies
            with tf.compat.v1.variable_scope('VS'):
                c_init_VS = np.zeros((1, self.num_LSTMunits_VS), np.float32)
                h_init_VS = np.zeros((1, self.num_LSTMunits_VS), np.float32)
                self.state_init_VS   = [c_init_VS, h_init_VS]            
                c_in_VS = tf.compat.v1.placeholder(tf.float32, [1, self.num_LSTMunits_VS])
                h_in_VS = tf.compat.v1.placeholder(tf.float32, [1, self.num_LSTMunits_VS])
                self.state_in_VS = [c_in_VS,h_in_VS]
                c_in_batch_VS=tf.tile(c_in_VS,[self.batch_size,1]) #shape (batch_size,lstm_cell.state_size.c)
                h_in_batch_VS=tf.tile(h_in_VS,[self.batch_size,1]) 
                self.state_in_batch_VS = [c_in_batch_VS,h_in_batch_VS]     
                state_in_VS = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_in_batch_VS, h_in_batch_VS) 
                self.lstm_cell_VS = tf.compat.v1.nn.rnn_cell.LSTMCell(self.num_LSTMunits_VS,state_is_tuple=True) 
                lstm_outputs_VS, lstm_state_VS = tf.compat.v1.nn.dynamic_rnn(
                self.lstm_cell_VS, inputs, initial_state=state_in_VS, sequence_length=self.seq_length, time_major=False)
                lstm_c_VS, lstm_h_VS = lstm_state_VS
                self.state_out_VS = [lstm_c_VS, lstm_h_VS] #this output will be feed in later, so should be the same shape as the self.state_in
                self.rnn_out_VS   = lstm_outputs_VS 
                lstm_outputs_VS_flatten=tf.reshape(lstm_outputs_VS, [-1, self.num_LSTMunits_VS])

            
            lstm_outputs = lstm_outputs_VS_flatten
            self.policy = tf.keras.layers.Dense(a_size, 
                activation=tf.nn.softmax,
                kernel_initializer=normalized_columns_initializer(0.01))(lstm_outputs)
            self.value_VS = tf.keras.layers.Dense(1, 
                activation=None,
                kernel_initializer=normalized_columns_initializer(1.0))(lstm_outputs_VS_flatten)
            
            self.actions = tf.compat.v1.placeholder(shape=[None],dtype=tf.int32,name="actions")
            self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v_VS  = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32) 
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                self.pi_as = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32,name="pi_a")
            
                #Loss functions
                self.value_loss_VS  = 0.5 * tf.reduce_sum(tf.square(self.target_v_VS  - tf.squeeze(self.value_VS,1)))
                self.value_loss = self.value_loss_VS/tf.cast(self.seq_length, tf.float32)

                self.entropy = - tf.reduce_sum(self.policy * tf.math.log(self.policy + 1e-7))/tf.cast(self.seq_length, tf.float32)
                ratio=self.responsible_outputs/(self.pi_as + 1e-7)
                surr1 = ratio * self.advantages
                surr2 = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon) * self.advantages
                surr  = tf.minimum(surr1, surr2)
                self.policy_loss = -tf.reduce_sum(surr)/tf.cast(self.seq_length, tf.float32)
                self.loss = self.policy_loss + beta_v *self.value_loss - self.entropy * beta_e
                self.loss /= tf.cast(self.batch_size, tf.float32)
                                
                #gradients
                local_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.linalg.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,10.0)
                global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

    def perform_trial(self,sess,obs,env,rnn_state_init_VS):
        trial_buffer = []
        trial_state_VS   = []
        done_trial=False
        trial_len = env.trial_len
        while done_trial==False: 
            a_dist,v_VS, rnn_state_VS,rnn_output_VS = \
            sess.run([self.policy,self.value_VS,
                      self.state_out_VS,self.rnn_out_VS], 
                feed_dict={
                self.batch_size: 1,
                self.seq_length: [1],
                self.obs:[[obs]],
                self.state_in_VS[0]: rnn_state_init_VS[0],
                self.state_in_VS[1]: rnn_state_init_VS[1]})    
            
            a = self.rng.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist[0] == a)
            pi_a=a_dist[0][a] # the prob for choosing action a, used as denominator in PPO
                             
            next_obs, reward, done_trial, info = env.step(a) 

            trial_buffer_temp=[a,reward]
            trial_buffer_temp.extend(obs)
            trial_buffer_temp.extend([v_VS[0,0],pi_a])
            trial_buffer.append(trial_buffer_temp) 
            trial_state_VS.append(rnn_state_VS)
            rnn_state_init_VS  = rnn_state_VS #within a trial, continous training
            obs = next_obs
        time_step_food = env.time_step_food_stored   
        trial_reward   = env.trial_reward_stored 
        return obs,time_step_food,trial_len,trial_reward,trial_buffer,\
               trial_state_VS,rnn_state_VS,v_VS[0,0]

class Worker():
    def __init__(self,np_seed,env,name,a_size,batch_size,
        beta_e,trainer,model_path,global_steps):
        self.seed = np_seed
        self.env  = env
        self.name = "worker_" + str(name)
        self.number       = name 
        self.time_horizon = time_horizon
        self.batch_size   = batch_size
        self.beta_e       = beta_e
        self.gamma_VS     = gamma_VS
        self.model_path   = model_path
        self.trainer      = trainer
        self.global_steps = global_steps
        self.increment    = self.global_steps.assign_add(1)
        self.summary_writer = tf.compat.v1.summary.FileWriter("train_"+str(self.number))
        self.local_AC = AC_Network(np_seed,self.beta_e,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
                
    def train(self,sess,rollout,batch_rnn_state_VS,reward_rate,value_rate):
        batch_rnn_state0_VS =np.squeeze(np.array(batch_rnn_state_VS[0]),axis=1)
        batch_rnn_state1_VS =np.squeeze(np.array(batch_rnn_state_VS[1]),axis=1)
        
        feature_end = 2+feature_dim+background_dim
        rollout     = np.array([trial[:self.time_horizon+1] for trial in rollout])
        
        values_VS   = rollout[:,:,feature_end+1]
        rollout     = rollout[:,:self.time_horizon,:]
        actions     = rollout[:,:,0]   
        rewards     = rollout[:,:,1]     
        obs         = rollout[:,:,2:feature_end+1]
        pi_as       = rollout[:,:,feature_end+2]
        
        reward_rate = (1 - lr_eta) * reward_rate + \
                      lr_eta * np.mean(rewards)
        value_rate  = (1 - lr_eta) * value_rate + \
                      lr_eta * np.mean(values_VS)

        expected_rewards_VS =np.zeros_like(actions)
        advantages_VS =np.zeros_like(actions)
        #advantages    =np.zeros_like(actions)
        for k in range(self.batch_size): 
            
            expected_rewards_VS[k] = rewards[k] - reward_rate + \
                                     self.gamma_VS * values_VS[k,1:self.time_horizon+1] - \
                                     rm_vbias_coeff * value_rate            
            # for VS
            advantages_VS[k] = rewards[k,:self.time_horizon] - reward_rate + \
                               self.gamma_VS * values_VS[k,1:self.time_horizon+1] - \
                               values_VS[k,:self.time_horizon] 
            advantages_VS[k] = discount(advantages_VS[k],lambda_gae)

        trial_lengths = self.time_horizon  
        feed_dict = {self.local_AC.batch_size: self.batch_size,
            self.local_AC.obs:obs,
            self.local_AC.actions:actions.flatten(),
            self.local_AC.pi_as:pi_as.flatten(),
            self.local_AC.target_v_VS:expected_rewards_VS.flatten(),            
            self.local_AC.advantages:advantages_VS.flatten(),
            self.local_AC.seq_length: [trial_lengths],         
            self.local_AC.state_in_batch_VS[0]: batch_rnn_state0_VS,
            self.local_AC.state_in_batch_VS[1]: batch_rnn_state1_VS}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return reward_rate,value_rate,v_l/self.batch_size, p_l/self.batch_size, e_l/self.batch_size, g_n,v_n
   
    def perform_batch(self,sess,obs,rnn_state_init_VS,reward_rate,value_rate):
        trial_lengths = []
        trial_rewards = []
        trial_values_VS  = []
        batch_buffer     = []
        batch_rnn_state_VS  =[]
        batch_rnn_state0_VS =[]
        batch_rnn_state1_VS =[]
        done_batch     = False
        batch_count    = 0
        
        while done_batch == False:
            trial_buffer_one_h = []

            batch_rnn_state0_VS.append(rnn_state_init_VS[0])
            batch_rnn_state1_VS.append(rnn_state_init_VS[1])
            
            # to collect data for one time horizon
            step = 0
            while step <= self.time_horizon:                           
                obs,_,trial_length,trial_reward,trial_buffer,\
                _,rnn_state_VS,value_VS \
                = self.local_AC.perform_trial(sess,obs,self.env,\
                    rnn_state_init_VS)                
                
                trial_buffer_one_h  += trial_buffer
                trial_lengths.append(trial_length)           
                
                step += trial_length
                
                rnn_state_init_VS  = rnn_state_VS 
                
            batch_buffer.append(trial_buffer_one_h)
            if batch_count == self.batch_size - 1:
                done_batch = True
        batch_rnn_state_VS =[batch_rnn_state0_VS,batch_rnn_state1_VS]
        reward_rate,value_rate,v_l,p_l,e_l,g_n,v_n = self.train(sess,batch_buffer,batch_rnn_state_VS,reward_rate,value_rate)
        return obs,trial_lengths,rnn_state_VS,v_l,p_l,e_l,g_n,v_n,reward_rate,value_rate
    def work(self,sess,coord,saver,train,traininfofile): 
        batch_count = sess.run(self.global_steps) + 1
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default(): 
            rnn_state_VS  = self.local_AC.state_init_VS 
            obs = self.env.reset()  
            reward_rate = 0
            value_rate = 0                     
            with open(traininfofile, 'w') as f:
                while not coord.should_stop():
                    
                    sess.run(self.update_local_ops)   
                    rnn_state_init_VS = rnn_state_VS                                     
                    obs,_,rnn_state_VS,v_l,p_l,e_l,g_n,v_n,reward_rate,value_rate = \
                    self.perform_batch(sess,obs,rnn_state_init_VS,reward_rate,value_rate)
                    
                    if batch_count % save_per_steps == 0:
                        if self.name == 'worker_0':
                            saver.save(sess,self.model_path+'/model-'+str(batch_count)+'.cptk')
                            f.write("Saved Model: %d\n" % batch_count)                   

                    # for every update step 
                    summary = tf.compat.v1.Summary()   
                    if train == True:
                        summary.value.add(tag='charts/reward_rate', simple_value=float(reward_rate))
                        summary.value.add(tag='charts/value_rate', simple_value=float(value_rate))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(beta_v*v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/loss', simple_value=float(beta_v*v_l+p_l-self.beta_e*e_l))
                        summary.value.add(tag='Losses/actual Entropy', simple_value=float(self.beta_e*e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, batch_count)
                    self.summary_writer.flush()    
                    f.flush()
                    if self.name == 'worker_0':
                        sess.run(self.increment)
                    batch_count += 1
                    if batch_count >= pretrain_steps:
                        done_pretrain = True
                        self.env.done_pretrain = done_pretrain                    
                    if batch_count > simulation_steps + pretrain_steps:
                        coord.request_stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed","-s", type=int, default=1,
                    help="seed for random number generator")   
    args = parser.parse_args()
    tf.compat.v1.reset_default_graph()
    
    seed = args.seed
    print("seed ", seed)
    np_seed = 2*seed
    tf_seed = 2*seed +1
    # Seed the TF random number generator for reproducible initialization
    tf.compat.v1.set_random_seed(tf_seed)
    traininfofile = "pretrain_info.txt"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with tf.device("/cpu:0"): 
        global_steps = tf.Variable(0,dtype=tf.int32,name='global_steps',trainable=False)
        trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        master_network = AC_Network(np_seed,beta_e,a_size,'global',None) # Generate global network
        #num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        num_workers = 1
        workers     = []
        for i in range(num_workers):
            workers.append(Worker(np_seed,cue_input(np_seed),i,a_size,batch_size,
                beta_e,trainer,model_path,global_steps))
        saver = tf.compat.v1.train.Saver(max_to_keep=None)

    with tf.compat.v1.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess,coord,saver,train,traininfofile)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            worker_threads.append(thread)    
        coord.join(worker_threads)

