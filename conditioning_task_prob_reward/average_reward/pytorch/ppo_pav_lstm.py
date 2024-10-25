import params
import argparse
import os
import time
from distutils.util import strtobool
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


num_envs = 1 # one session
simulation_steps = params.simulation_steps
pretrain_steps   = params.pretrain_steps
save_per_steps   = params.save_per_steps
time_horizon     = params.time_horizon
num_steps = time_horizon
dt = params.dt

feature_dim      = params.feature_dim
background_dim   = params.background_dim
click_dim        = params.click_dim
click_strength   = params.click_strength
poking_cost      = params.poking_cost
trial_shift      = params.trial_shift
ITI_min          = params.ITI_min
ITI_max          = params.ITI_max
reward_delay     = params.reward_delay
holding_time     = params.holding_time
cue_duration_aud = params.cue_duration_aud
cue_duration_rew = params.cue_duration_rew

lr               = params.lr
beta_e           = params.beta_e
beta_v           = params.beta_v
epsilon          = params.epsilon
gae_lambda       = params.lambda_gae
action_names     = params.action_names
a_size           = params.a_size
load_model       = False
model_path       = './model_meta'

lr_eta = 0.1
rm_vbias_coeff = 0.5

obs_dim  = feature_dim + background_dim + click_dim
num_updates = simulation_steps + pretrain_steps

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
                reward = 1.0
            else:
                reward = 0.0   
        elif self.cue_type == '1':
            if self.rng.uniform() < 0.75:
                reward = 1.0
            else:
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
        
        self.time_step+=1            
        self.pre_a = a         
        info = {}        

        if not done_trial:
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

def normalized_columns_initializer_(tensor, std=1.0):
    out = np.random.randn(*tensor.size()).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=1, keepdims=True))
    with torch.no_grad():
        tensor.copy_(torch.from_numpy(out))

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.num_LSTMunits=32 
        self.input_feature = feature_dim + background_dim + click_dim
        self.lstm = nn.LSTM(self.input_feature, self.num_LSTMunits)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # input-hidden weights
                nn.init.xavier_uniform_(param.data)  # Glorot uniform initializer
            elif 'weight_hh' in name:  # hidden-hidden weights
                nn.init.xavier_uniform_(param.data)  # Glorot uniform initializer
            elif 'bias_ih' in name:  # input-hidden bias
                nn.init.constant_(param.data, 0)  # Initialize biases to zero
            elif 'bias_hh' in name:  # hidden-hidden bias
                param.data.fill_(0)  # Initialize biases to zero
                # Set the forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)  # Initialize forget gate bias to 1

        # Modify actor initialization
        self.actor = nn.Linear(self.num_LSTMunits, len(action_names))
        normalized_columns_initializer_(self.actor.weight, std=0.01)
        nn.init.constant_(self.actor.bias, 0)
        
        # Modify critic initialization
        self.critic = nn.Linear(self.num_LSTMunits, 1)
        normalized_columns_initializer_(self.critic.weight, std=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def get_states(self, x, lstm_state):
        hidden = x 
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        new_hidden, lstm_state = self.lstm(hidden, lstm_state)
        new_hidden = torch.flatten(new_hidden, 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state):
        hidden, _ = self.get_states(x, lstm_state)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, action = None):
        hidden, lstm_state = self.get_states(x, lstm_state)
        logits = self.actor(hidden)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        actions_onehot = F.one_hot(action, num_classes=a_size)
        selected_probs = torch.sum(probs * actions_onehot.float(), axis=-1)
        return action,selected_probs,dist.entropy(),self.critic(hidden),probs,lstm_state

if __name__ == "__main__":
    traininfofile = "pretrain_info.txt"
    paramlogfile  = "param_info.txt"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed","-s", type=int, default=1,
                    help="seed for random number generator")   
    args = parser.parse_args()

    with open("paramlogfile", "a") as f:
        f.write("seed of noise: %d\n" % args.seed)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    np_seed = 2*args.seed
    pytorch_seed = 2*args.seed +1
    np.random.seed(np_seed)
    torch.manual_seed(pytorch_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    pretrain = False
    envs = cue_input(np_seed)

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    obs      = torch.zeros((num_steps, num_envs,obs_dim)).to(device)
    actions  = torch.zeros((num_steps, num_envs)).to(device)
    pi_as = torch.zeros((num_steps, num_envs)).to(device)
    rewards  = torch.zeros((num_steps, num_envs)).to(device)
    values   = torch.zeros((num_steps, num_envs)).to(device)
    reward_rate = torch.zeros(num_envs).to(device)
    value_rate  = torch.zeros(num_envs).to(device)
    
    next_obs   = torch.Tensor(envs.reset()).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
    )  
    for update in range(1, num_updates + 1):
               
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        step = 0
        done = False
        while not (step>num_steps and done):
            with torch.no_grad():
                action,prob,_,value,_,next_lstm_state = \
                    agent.get_action_and_value(next_obs, next_lstm_state)
            if step < num_steps:
                obs[step] = next_obs                   
                actions[step]  = action
                pi_as[step]  = prob
                values[step] = value.flatten()
            if step == num_steps:
                next_value = value.flatten()
            next_obs, reward, done, info = envs.step(action.cpu().numpy()[0])
            if step < num_steps:
                rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            step += 1
        with torch.no_grad():
            reward_rate = (1 - lr_eta) * reward_rate + \
                lr_eta * rewards.mean(dim=0)
            value_rate = (1 - lr_eta) * value_rate + \
                lr_eta * values.mean(dim=0)

            advantages = torch.zeros_like(rewards).to(device)
            returns = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = values[t + 1]
                delta = rewards[t] - reward_rate + nextvalues - values[t]
                advantages[t] = lastgaelam = delta + gae_lambda * lastgaelam
                returns[t] = rewards[t] - reward_rate + nextvalues

        # flatten the batch
        b_obs   = obs.reshape(-1,obs_dim)
        b_probs = pi_as.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_value_rate  = value_rate
        _,newprob,entropy,newvalue,_,_ = agent.get_action_and_value(
            b_obs,
            initial_lstm_state,
            b_actions.long(),
        )
        ratio = newprob/(b_probs + 1e-7)
       
        # Policy loss
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        v_loss = 0.5 * ((b_returns - rm_vbias_coeff * b_value_rate - newvalue) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - beta_e * entropy_loss + v_loss * beta_v
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 10)
        optimizer.step()
        if update % save_per_steps == 0:
            
            print("update: ", update)
            PATH = model_path+'/model-'+str(update)+'.pt'
            torch.save({
                'update': update,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
            
        if update >= pretrain_steps:
            envs.done_pretrain = True
