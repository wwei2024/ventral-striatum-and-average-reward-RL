import params
import argparse
import os
import time
from distutils.util import strtobool
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

d_state  = 0 # states  are continuous
d_return = 0 # returns are continuous
model_path = './model_meta'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=2e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--lr-eta", type=float, default=0.1,
        help="the learning rate of the average reward and value")
    parser.add_argument("--rm-vbias-coeff", type=float, default=0.5,
        help="coef of mean value bias")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=20000000000,
        help="total timesteps of the experiments")
    
    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=400,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.05,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=10.0,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


dt = params.dt # sec
ITI_min     = params.ITI_min
ITI_max     = params.ITI_max # sec
holding_min = params.holding_min
holding_max = params.holding_max
go_cue_duration  = params.go_cue_duration
rew_cue_duration = params.rew_cue_duration
click_strength   = params.click_strength

action_cost    = params.action_cost
switch_latency = params.switch_latency
action_names = params.action_names
port_names   = params.port_names
background_dim = 3

class banditEnv(gym.Env):    
    def __init__(self,block_probs=None,pretrain=False):
        super(banditEnv, self).__init__()

        if block_probs==None:
            self.block_probs=[[0.1,0.1],[0.1,0.5],[0.5,0.1],[0.1,0.9],
            [0.9,0.1],[0.5,0.9],[0.9,0.5],[0.5,0.5],[0.9,0.9]]
        else:
            self.block_probs = block_probs
        self.pretrain = pretrain
        self.num_actions  = len(action_names) # 7
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(low=-1, high=5,
                                        shape=(13+background_dim,), dtype=np.float32)
        self.central_ports_onehot = [[0, 1, 0]]
        self.nb_ports             = [[1, 0, 1]]
        self.go_cue_duration  = go_cue_duration
        self.rew_cue_duration = rew_cue_duration 
        self.click_strength   = click_strength
        self.background_dim   = background_dim # the same as in conditioning task
        self.switch_latency   = switch_latency
        self.action_cost      = action_cost

        cue_type=0
        self.central_port_onehot=self.central_ports_onehot[cue_type]
        self.nb_port_onehot=self.nb_ports[cue_type] 
        self.central_port_in = action_names[cue_type+1]
        self.left_port_in = action_names[cue_type]
        self.right_port_in = action_names[cue_type+2]
        self.choosing_options=[self.left_port_in,self.right_port_in]

        self.pre_a = action_names.index('non_engage')
        self.pre_a_s = action_names.index('non_engage')
        self.pre_r = 0.0

        if self.pretrain:
            self.done_learn_CNI = True
            self.done_learn_choosing = True
            self.done_learn_food_port = True
            self.done_explore_bandits = True
            self.done_adding_route_quit = True
            self.done_pre_train = True
        else:
            self.done_learn_CNI = False
            self.done_learn_choosing = False
            self.done_learn_food_port = False
            self.done_explore_bandits = False
            self.done_adding_route_quit = False
            self.done_pre_train = False
            
    def bandit_prob_reset(self):
        self.blockstep = 0     
        block=self.np_random.integers(len(self.block_probs))
        self.bandit = self.block_probs[block]
        self.blocklength=40+self.np_random.integers(20)  #40-60 block length

    def reset(self,seed=None):  
        super().reset(seed=seed)            
        self.time_step = 0  
        self.CNI         = False
        self.side_in     = False # True after 0-1 or 0-2
        self.click_on    = False
        self.action_switching_step = 0
        self.reward_to_be = 0
        self.reg_to_be    = 0
        self.choice       = np.nan

        if not self.done_learn_food_port:
            self.ITI_time = 2
            self.holding_time = 2  
            self.pre_a = action_names.index('non_engage')
            self.pre_a_s = action_names.index('non_engage')
            self.pre_r = 0.0
            self.bandit = self.block_probs[-1]
        else:            
            if holding_max>holding_min:
                self.holding_time = self.np_random.integers(holding_min,holding_max+1)
            else:
                self.holding_time = holding_min            
            self.ITI_time = self.np_random.integers(ITI_min,ITI_max)               

        if self.done_explore_bandits:  
            self.add_route_quit = True
        else:
            self.add_route_quit = False

        self.time_step_light_on = self.ITI_time          
        self.max_time_steps_CNI = self.ITI_time + 20
        self.max_time_steps     = self.max_time_steps_CNI+10

        pre_a_onehot = list(np.eye(self.num_actions)[self.pre_a])
        obs = [0,0,0]+[0,0]+pre_a_onehot+[self.pre_r]+[1.0]*self.background_dim
        return np.array(obs).astype(np.float32)
        
    def pullArm(self,action):
        if action==self.left_port_in: 
            action_num = 0
        elif action==self.right_port_in: 
            action_num = 1
        else:
            print('has to choose left or right!')
            return -1
        bandit = self.bandit[action_num]
        if bandit<self.bandit[1-action_num]:
            regret=self.bandit[1-action_num]-bandit            
        else:
            regret=0
                         
        if self.np_random.uniform() < bandit:
            reward = 1
        else:
            reward = 0 
        return reward,regret 
    def update_env(self):
        if self.blockstep==self.blocklength-1:
            self.bandit_prob_reset()
        else:
            self.blockstep += 1

    def step(self,a):        
        a_name       = action_names[a]
        pre_a_name   = action_names[self.pre_a]
        pre_a_s_name = action_names[self.pre_a_s]
        failed_trial = False
        done_trial   = False

        if self.time_step > self.max_time_steps-1:
            failed_trial = True

        if (self.time_step < 1) and (a_name not in ['non_engage','transition']): # error: starting a new trial, not through choosing lighted central port
            failed_trial = True
            
        if (self.time_step > 0) and (a_name!=pre_a_name) and (a_name!='transition' and pre_a_name!='transition'): # error: transition not through the switching action 'transition'
            failed_trial = True
             
        if (self.time_step < self.time_step_light_on) and (a_name == self.central_port_in): 
        # error: too early for CNI. only choosing 'non-engage' or 'engage' during ITI
            failed_trial = True
        
        # too early for poking into left, right, food ports.   
        if (self.CNI==False) and (a_name in self.choosing_options \
        or a_name=='food_port' or self.time_step>=self.max_time_steps_CNI): 
            failed_trial = True
        
        # not keep holding   
        if (self.CNI==True) and (self.time_step >= self.time_step_CNI) and \
        (self.time_step < self.time_step_go_cue) and (a_name!=self.central_port_in):
            failed_trial = True
            
        if (self.CNI==True) and (self.time_step >= self.time_step_go_cue) and (self.side_in==False) \
        and ((a_name in ['non_engage','engage','food_port']) or (pre_a_name=='transition' and a_name==self.central_port_in)):  
            failed_trial = True
 
        if (self.side_in==True) and (a_name==self.central_port_in or a_name=='engage' \
            or (pre_a_name==self.left_port_in and (a_name!=self.left_port_in and a_name!='transition')) \
            or (pre_a_name==self.right_port_in and (a_name!=self.right_port_in and a_name!='transition'))):
            failed_trial = True
           
        if failed_trial == True: 
            reward = -1 
            done_trial=True

        if done_trial==False:                                
            if a_name in ['non_engage']:
                reward = 0.0
            else:
                reward = self.action_cost

            if (a_name != pre_a_name) and self.time_step>0: # donot compare the action at time 0 with action from last trial                    
                
                if a_name=='transition': # start action switch
                    self.pre_a_s = self.pre_a
                    self.action_switching_step = 1
                    
                else: # finish action switch
                    if (a_name==pre_a_s_name) \
                    or (a_name==self.central_port_in and pre_a_s_name=='non_engage') \
                    or (a_name==self.left_port_in    and pre_a_s_name==self.right_port_in) \
                    or (a_name==self.right_port_in   and pre_a_s_name==self.left_port_in) \
                    or (a_name=='food_port' and pre_a_s_name==self.central_port_in) \
                    or ((a_name=='non_engage') and (pre_a_s_name in self.choosing_options) and (self.add_route_quit==False)):
                    # avoid self-switch, CNI from non-engage 
                        reward=-1
                        failed_trial=True
                        done_trial  = True                                                      
                    else:
                        if self.action_switching_step<self.switch_latency-1: # switch too early. should be >= switch_latency-1 
                            reward=-1
                            failed_trial=True
                            done_trial = True

                        elif (self.CNI==False) and (a_name=='non_engage' or a_name=='engage'):
                            pass                                  

                        elif (self.CNI==False) and (pre_a_s_name=='engage') \
                        and (a_name==self.central_port_in) and (self.time_step>=self.time_step_light_on):                                                
                            self.CNI=True 
                            self.time_step_CNI    = self.time_step
                            self.time_step_go_cue = self.holding_time+self.time_step_CNI
                            if not self.done_learn_CNI:
                                reward = 0.2
                                done_trial = True                            
                                    
                        elif (self.CNI==True) and (a_name in self.choosing_options) and (self.time_step >= self.time_step_go_cue): 
                            self.time_step_side_in = self.time_step 
                            self.side_in = True                                                                 
                            self.reward_to_be,self.reg_to_be = self.pullArm(a_name) 
                            if a_name == 'port_0':
                                self.choice = 1.0
                            else:
                                self.choice = 0.0
                            if not self.done_learn_choosing:
                                reward = 0.5
                                reg = 0
                                done_trial = True
                            else:
                                if self.reward_to_be > 0.5: # reward_to_be = 1
                                    self.click_on = True
                                       
                        elif (self.side_in==True) and (a_name == 'food_port') and (pre_a_s_name in self.choosing_options): 
                            # food port in
                            self.time_step_food = self.time_step                                
                            reward = self.reward_to_be
                            reg = self.reg_to_be
                            done_trial = True

                        elif (self.side_in==True) and (a_name=='non_engage') and (pre_a_s_name in self.choosing_options): 
                            # finish a non-rewarded trial or give up in a rewarded trial                           
                            #reg = self.reward_to_be # since no food picking
                            done_trial = True

                        else:
                            reward=-1
                            failed_trial=True
                            print("unknown error: ",a,reward,self.pre_a,self.pre_r,self.pre_a_s, self.CNI, self.side_in)
                            #reg=max(env.bandit)+1
                            done_trial = True
                                                        
            else: #here either time_step == 0 or a == pre_a 
                if a_name == 'transition':
                    self.action_switching_step += 1 
                else:
                    pass
        self.pre_a = a 
        self.pre_r = reward

        if done_trial and (not failed_trial):
            self.update_env()
        
        info = {}        
        info['failed_trial'] = failed_trial
        info['choice_trial'] = self.choice

        if not done_trial:
            self.time_step+=1
            light_on = [0.0, 0.0, 0.0]
            go_cue     = 0.0 
            reward_cue = 0.0            
            if self.CNI == False and self.time_step >= self.time_step_light_on:
                light_on  = self.central_port_onehot                                   
            if self.CNI == True: 
                if self.time_step >= self.time_step_light_on and \
                self.time_step < self.time_step_go_cue:
                    light_on  = self.central_port_onehot
                else:
                    light_on  = self.nb_port_onehot
                if self.time_step >= self.time_step_go_cue and \
                self.time_step < self.time_step_go_cue + go_cue_duration:
                    go_cue    = 1.0            
            if self.click_on == True: 
                reward_cue = self.click_strength
                self.click_on = False
                  
            obs = light_on + [go_cue] + [reward_cue] + \
                  list(np.eye(self.num_actions)[a]) + [reward]+[1.0]*self.background_dim
        else:
            obs = self.reset()        
        return np.array(obs).astype(np.float32), reward, done_trial, info

def make_env(seed,pretrain):
    def thunk():
        env = banditEnv(pretrain = pretrain)
        env.reset(seed = seed)
        env.bandit_prob_reset() #this will assign env.bandit
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.num_LSTMunits=64 
        self.input_feature = 13+background_dim
        self.lstm = nn.LSTM(self.input_feature, self.num_LSTMunits)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor  = layer_init(nn.Linear(self.num_LSTMunits, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(self.num_LSTMunits, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = x 
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action,probs.log_prob(action),probs.entropy(),self.critic(hidden),probs,lstm_state


if __name__ == "__main__":
    traininfofile = "pretrain_info.txt"
    paramlogfile  = "param_info.txt"
    args = parse_args()
    with open("paramlogfile", "a") as f:
        f.write("seed of noise: %d\n" % args.seed)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter(f"runs")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    pretrain = False
    if not pretrain:
        pretrain_full=False
    pretrain_finished = False

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i,pretrain) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-4)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  
    num_updates = args.total_timesteps // args.batch_size
    print("num_updates:", num_updates)

    avg_win = 100
    fail_th = 0.25
    batch_duration = 40 
    num=5    
    maxlen = avg_win*4
    failing_rates    = deque(maxlen=maxlen) #here maxlen is trial numbers
    rewards_perstep  = deque(maxlen=maxlen)
    rewards_pertrial = deque(maxlen=maxlen) #here maxlen is non-error trial numbers
    choices_pertrial = deque(maxlen=maxlen)

    rewards_pertrial_long = deque(maxlen=5000)
    
    reward_rate = torch.zeros(args.num_envs).to(device)
    value_rate  = torch.zeros(args.num_envs).to(device)
    with open(traininfofile, 'a') as f:

        for update in range(1, num_updates + 1):

            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                with torch.no_grad():
                    action, logprob, _, value, _, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done*d_state)
                    values[step] = value.flatten()
                actions[step]  = action
                logprobs[step] = logprob
                
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
                
                
                # this must have bracket and must use & but not and
                fail_info = info['failed_trial']
                choice_info   = info['choice_trial']
                choices_pertrial.extend(choice_info[(choice_info>=0) & done])
                if not pretrain_finished:
                    rew_t = reward[done]
                else:
                    rew_t = reward[(fail_info<1) & done]
                rewards_pertrial.extend(rew_t) 
                rewards_pertrial_long.extend(rew_t)
                failing_rates.extend(fail_info[done])
                
            rewards_perstep.append(np.mean(rewards.cpu().numpy()))
                
            #bootstrap value if not done
            with torch.no_grad():
                reward_rate = (1 - args.lr_eta) * reward_rate + \
                    args.lr_eta * rewards.mean(dim=0)
                value_rate = (1 - args.lr_eta) * value_rate + \
                    args.lr_eta * values.mean(dim=0)

                next_value = agent.get_value(
                    next_obs,
                    next_lstm_state,
                    next_done,
                ).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done * d_return
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1] * d_return
                            nextvalues = values[t + 1]
                        delta = rewards[t] - reward_rate + nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done * d_return
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1] * d_return
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] - reward_rate + nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_dones = dones.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_value_rate  = value_rate.repeat_interleave(args.num_steps)
            # Optimizing the policy and value network
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
            clipfracs = []
            for epoch in range(args.update_epochs):
                rng.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()
                    _, newlogprob, entropy, newvalue, _, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                        b_dones[mb_inds]*d_state,
                        b_actions.long()[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((b_returns[mb_inds] - args.rm_vbias_coeff * b_value_rate[mb_inds] - 
                        newvalue) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            if update % avg_win == 0:
                if update % 1000 == 0:
                    print("update: ", update)
                    f.write(f"Saved Model: {update}, {global_step}\n")
                if update % 1000 == 0:
                    PATH = model_path+'/model-'+str(update)+'.pt'
                    torch.save({
                        'update': update,
                        'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, PATH)

                failing_rate   = np.mean(failing_rates)
                rewards_report = np.mean(rewards_perstep)   
                rewards_pertrial_report = np.mean(rewards_pertrial)  
                choice_bias = np.mean(choices_pertrial)       

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar("charts/failing", failing_rate, global_step)
                writer.add_scalar("charts/reward_perstep", rewards_report, global_step)
                writer.add_scalar("charts/reward_pertrial", rewards_pertrial_report, global_step)
                writer.add_scalar("charts/choices_bias", choice_bias, global_step)
                writer.add_scalar("charts/reward_rate", reward_rate.mean().item(), global_step)
                writer.add_scalar("charts/value_bias", value_rate.mean().item(), global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                
                batch_count = update 

                done_learn_CNI       = envs.get_attr(name='done_learn_CNI') 
                done_learn_choosing  = envs.get_attr(name='done_learn_choosing')
                done_learn_food_port = envs.get_attr(name='done_learn_food_port')
                done_explore_bandits = envs.get_attr(name='done_explore_bandits')
                done_adding_route_quit = envs.get_attr(name='done_adding_route_quit')
                done_pre_train = envs.get_attr(name='done_pre_train') 

                if not all(done_pre_train):
                    if not all(done_learn_CNI):
                        if failing_rate<fail_th:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_learn_CNI',values=True)
                            print("batch_count_l2_learn_CNI: ", batch_current_learn_stage)
                            f.write("batch_count_l2_learn_CNI: %d\n" % batch_current_learn_stage)
                    if all(done_learn_CNI) and (batch_count > batch_current_learn_stage) and not all(done_learn_choosing): 
                        if failing_rate>1-fail_th and batch_count > batch_current_learn_stage + num*batch_duration:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_learn_CNI',values=False)
                            print("batch_count_l2_CNI_F: ", batch_count)
                            f.write("batch_count_l2_CNI_F: %d\n" % batch_count)
                        if failing_rate<fail_th:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_learn_choosing',values=True)
                            print("batch_count_l2_learn_choosing: ", batch_current_learn_stage)
                            f.write("batch_count_l2_learn_choosing: %d\n" % batch_current_learn_stage)
                    if all(done_learn_choosing) and (batch_count > batch_current_learn_stage) and not all(done_learn_food_port): 
                        if failing_rate>1-fail_th and batch_count > batch_current_learn_stage + num*batch_duration:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_learn_choosing',values=False)
                            print("batch_count_l2_choosing_F: ", batch_count)
                            f.write("batch_count_l2_choosing_F: %d\n" % batch_count)
                        if failing_rate<fail_th:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_learn_food_port',values=True)
                            print("batch_count_l2_learn_food_port: ", batch_current_learn_stage)
                            f.write("batch_count_l2_learn_food_port: %d\n" % batch_current_learn_stage)
                    if all(done_learn_food_port) and (batch_count > batch_current_learn_stage) and not all(done_explore_bandits):
                        
                        if failing_rate>1-fail_th and batch_count > batch_current_learn_stage + num*batch_duration:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_learn_food_port',values=False)
                            print("batch_count_l2_food_port_F: ", batch_count)
                            f.write("batch_count_l2_food_port_F: %d\n" % batch_count)
                        if failing_rate<fail_th and batch_count > batch_current_learn_stage + 10*batch_duration:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_explore_bandits',values=True)
                            print("batch_count_explore_bandits: ", batch_current_learn_stage)
                            f.write("batch_count_explore_bandits: %d\n" % batch_current_learn_stage)
                    if all(done_explore_bandits) and (batch_count > batch_current_learn_stage) and not all(done_adding_route_quit):        
                        
                        if failing_rate>1-fail_th and batch_count > batch_current_learn_stage + num*batch_duration:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_explore_bandits',values=False)
                            print("batch_count_explore_bandits_F: ", batch_count)
                            f.write("batch_count_explore_bandits_F: %d\n" % batch_count)
                        if failing_rate<fail_th:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_adding_route_quit',values=True)
                            print("batch_count_adding_route_quit: ", batch_current_learn_stage)
                            f.write("batch_count_adding_route_quit: %d\n" % batch_current_learn_stage)
                    if all(done_adding_route_quit) and (batch_count > batch_current_learn_stage) and not all(done_pre_train):
                        if failing_rate>1-fail_th and batch_count > batch_current_learn_stage + num*batch_duration:
                            batch_current_learn_stage = batch_count
                            envs.set_attr(name='done_adding_route_quit',values=False)
                            print("batch_count_adding_route_quit_F: ", batch_count)
                            f.write("batch_count_adding_route_quit_F: %d\n" % batch_count)
                        if failing_rate<fail_th:
                            batch_current_learn_stage = batch_count
                            pretrain_finished = True
                            batch_count_done_pretrain = batch_count
                            envs.set_attr(name='done_pre_train',values=True)
                            print("batch_count_pretrain: ", batch_current_learn_stage)
                            f.write("batch_count_pretrain: %d\n" % batch_current_learn_stage)
                else:
                    if failing_rate>1-fail_th and batch_count > batch_count_done_pretrain + num*batch_duration:
                        batch_current_learn_stage = batch_count
                        envs.set_attr(name='done_pre_train',values=False)
                        print("batch_count_pretrain_F: ", batch_count)
                        f.write("batch_count_pretrain_F: %d\n" % batch_count)                    
                f.flush()

    envs.close()
    writer.close()
