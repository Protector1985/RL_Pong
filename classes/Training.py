import gym
import torch.optim as optim
import torch
from torch.nn import functional as F
import numpy as np

class Training:
    
    def worker(self, t, worker_model, counter, params):
        worker_env = gym.make('ALE/Pong', render_mode="human")
        # worker_env.reset()
        worker_optimizer = optim.Adam(lr=1e-4, params=worker_model.parameters())
        worker_optimizer.zero_grad()
        
        for i in range(params['epochs']):
            worker_optimizer.zero_grad()
            
            #run_episode will play one episode of the game
            values, logprobs, rewards = self.run_episode(worker_env, worker_model)
            
            #update_params will take collected data from run_episode and update the parameters
            actor_loss, critic_loss, eplen = self.update_params(worker_optimizer, values, logprobs, rewards)
            
            counter.value = counter.value +1
            
    #runs a training episode
    def run_episode(self, worker_env, worker_model):
       
        observation = worker_env.reset()
        obs = np.transpose(observation[0], (2, 0, 1))
        # Convert to PyTorch tensor and add a batch dimension
        state = torch.from_numpy(obs).float().unsqueeze(0)

        #values=critic, logprobs=actor, 
        values, logprobs, rewards = [], [], []
        done = False
        j=0
        
        while (done == False):
            j += 1
            #policy=actor, value=critic
            policy, value = worker_model(state)
            values.append(value)
            logits = policy.view(-1)
            
            #retrieves the actions from the model output for actor
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            
            logprob_ = policy.view(-1)[action]
            logprobs.append(logprob_)
            
            observation_, reward, terminated, truncated, info = worker_env.step(action.detach().item()+2)
            
            observation = np.transpose(observation_, (2, 0, 1))
            state = torch.from_numpy(observation).float().unsqueeze(0)
            
            if terminated or truncated:
                reward = -10
                worker_env.reset()
            else:
                reward = 1.0
            rewards.append(reward)
            
        return value, logprobs, rewards
    
    def update_params(self, worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        
        ret_ = torch.Tensor([0])
        
        
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + gamma + ret_
            Returns.append(ret_)
            
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns, dim=0)
        
        actor_loss = -1 * logprobs * (Returns - values.detach())
        critic_loss = torch.pow(values - Returns, 2)
        
        loss = actor_loss.sum() + clc*critic_loss.sum()
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)
        
        