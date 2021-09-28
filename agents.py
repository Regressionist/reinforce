import gym
import torch
import torch.optim as optim
from torch.distributions import Categorical

from policies import MLPPolicy
from typing import Iterator
from typing import Tuple
from typing import List
from typing import Dict

class DavidAgent():
    def __init__(self, env_name: str):
        '''
        Args
            env_name: Gym environment to train the model on
        '''
        self.env = gym.make(env_name)
        self.observation_space_dim = self.env.observation_space._shape[0]
        self.action_space_dim = self.env.action_space.n
        self.policy = MLPPolicy(
            observation_space_dim=self.observation_space_dim,
            action_space_dim=self.action_space_dim
        )
        self.optimizer = optim.Adam(self.policy.parameters())
        self.max_steps = 100000
    
    def select_action(
            self,
            state: Iterator[float]) -> Tuple[int, float]:
        '''
        Given a `state`, samples an action using
        `self.policy`. Returns the action and the log probability
        of the action 
        '''
        state_tensor = torch.tensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def play_episode(self) -> Dict[str, List[float]]:
        '''
        Runs one episode of the environment
        using `self.policy`. Returns the log probabilities
        of all actions taken along with their corresponding
        rewards
        '''
        state = self.env.reset()
        episode = {
            'action_log_probs': [],
            'rewards': []
        }
        for t in range(0, self.max_steps):
            action, log_prob_action = self.select_action(state)
            episode['action_log_probs'].append(log_prob_action)
            state, reward, done, info = self.env.step(action)
            episode['rewards'].append(reward)
            if done:
                break
        return episode
    
    def update_policy(
            self,
            max_episodes: int,
            save_pth: str,
            logging: bool=True,
            save: bool=True,
            checkpoint: int=100) -> List[float]:
        '''
        Plays `max_episodes` number of episodes
        and after each episode, updates the 
        `self.policy` using the rewards generated for
        that episode and the actions taken. The policy update
        happens only after one complete episode has been
        played using the current `self.policy` and there's
        only one update happening for an entire episode
        Args:
            max_episodes: Total number of episodes to train
            save_pth: Save the best agent trained to this path
            checkpoint: number of episodes after which the running
                rewards are logged
            logging: True if you want to log the training rewards on screen
            save: True if you want to save the best model
        Returns:
        Return of each episode played
        '''
        training_returns = []
        running_reward = 0
        eps = 1e-05
        best_return = 0
        for ep_idx in range(max_episodes):
            self.optimizer.zero_grad()
            
            episode = self.play_episode()
            
            returns = []
            future_return = 0
            for r in episode['rewards'][::-1]:
                future_return += r
                returns.insert(0, future_return)

            training_returns.append(returns[0])
            if returns[0] >= best_return:
                best_return = returns[0]
                if save:
                    torch.save(self.policy, save_pth)
                           
            running_reward = running_reward * 0.9 +  returns[0] * 0.1
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) /  ((returns.std() + eps))
            log_probs = episode['action_log_probs']
            
            policy_loss = []
            for i in range(len(log_probs)):
                policy_loss.append(- returns[i] * log_probs[i])
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
            
            if logging and (ep_idx + 1) % checkpoint == 0:
                print(f"Episode: {ep_idx + 1} | score: {running_reward}")
        return training_returns


class SuttonAgent():
    def __init__(self, env_name: str):
        '''
        Args
            env_name: Gym environment to train the model on
        '''
        self.env = gym.make(env_name)
        self.observation_space_dim = self.env.observation_space._shape[0]
        self.action_space_dim = self.env.action_space.n
        self.policy = MLPPolicy(
            observation_space_dim=self.observation_space_dim,
            action_space_dim=self.action_space_dim
        )
        self.optimizer = optim.Adam(self.policy.parameters())
        self.max_steps = 100000
    
    def select_action(
            self,
            state: Iterator[float]) -> int:
        '''
        Given a `state`, samples an action using
        `self.policy`. Returns the action sampled
        '''
        state_tensor = torch.tensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        try:
            m = Categorical(action_probs)
        except:
            print(action_probs)
            m = Categorical(action_probs)
        action = m.sample()
        return action.item()
    
    def play_episode(self) -> Dict[str, List[float]]:
        '''
        Runs one episode of the environment
        using `self.policy`. Returns states, actions,
        and rewards observed during this episode
        '''
        self.policy.eval()
        with torch.no_grad():
            state = self.env.reset()
            episode = {
                'states': [],
                'actions': [],
                'rewards': []
            }
            for t in range(0, self.max_steps):
                episode['states'].append(state)
                action = self.select_action(state)
                episode['actions'].append(action)
                state, reward, done, info = self.env.step(action)
                episode['rewards'].append(reward)
                if done:
                    break
        return episode
    
    def update_policy(
            self,
            max_episodes: int,
            save_pth: str,
            save: bool=True,
            logging: bool=True,
            checkpoint: int=100) -> List[float]:
        '''
        Plays `max_episodes` number of episodes
        and after each episode, updates the 
        `self.policy` using the rewards generated for
        that episode and the actions taken. The policy update
        happens only after one complete episode has been
        played using the current `self.policy` and the number of
        updates done for any episode is equal to the length of the
        episode (each state, action, reward triplet corresponds to one
        update)
        Args:
            max_episodes: Total number of episodes to train
            save_pth: Save the best agent trained to this path
            checkpoint: number of episodes after which the running
                rewards are logged
            logging: True if you want to log the training rewards on screen
            save: True if you want to save the best model
        Returns:
        Return of each episode played
        '''
        training_returns = []
        running_reward = 0
        best_return = 0
        for ep_idx in range(max_episodes):
            
            episode = self.play_episode()
            self.policy.train()
            
            returns = []
            future_return = 0
            for r in episode['rewards'][::-1]:
                future_return += r
                returns.insert(0, future_return)

            training_returns.append(returns[0])
            if returns[0] >= best_return:
                best_return = returns[0]
                if save:
                    torch.save(self.policy, save_pth)
                           
            running_reward = running_reward * 0.9 +  returns[0] * 0.1
            actions = episode['actions']
            
            for i in range(len(actions)):
                self.optimizer.zero_grad()
                state = torch.tensor(episode['states'][i]).unsqueeze(0)
                probs = self.policy(state)[0]
                log_prob = torch.log(probs[actions[i]])
                loss = - returns[i] * log_prob
                loss.backward()
                self.optimizer.step()
            
            if logging and (ep_idx + 1) % checkpoint == 0:
                print(f"Episode: {ep_idx + 1} | score: {running_reward}")
        return training_returns


class SuttonAgentImportanceWeighted():
    def __init__(self, env_name: str):
        '''
        Args
            env_name: Gym environment to train the model on
        '''
        self.env = gym.make(env_name)
        self.observation_space_dim = self.env.observation_space._shape[0]
        self.action_space_dim = self.env.action_space.n
        self.policy = MLPPolicy(
            observation_space_dim=self.observation_space_dim,
            action_space_dim=self.action_space_dim
        )
        self.optimizer = optim.Adam(self.policy.parameters())
        self.max_steps = 100000
    
    def select_action(
            self,
            state: Iterator[float]) -> Tuple[int, float]:
        '''
        Given a `state`, samples an action using
        `self.policy`. Returns the action sampled
        '''
        state_tensor = torch.tensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        try:
            m = Categorical(action_probs)
        except:
            print(action_probs)
            m = Categorical(action_probs)
        action = m.sample()
        return action.item(), action_probs[0][action]
    
    def play_episode(self) -> Dict[str, List[float]]:
        '''
        Runs one episode of the environment
        using `self.policy`. Returns states, actions,
        and rewards observed during this episode
        '''
        self.policy.eval()
        with torch.no_grad():
            state = self.env.reset()
            episode = {
                'states': [],
                'actions': [],
                'probs' : [],
                'rewards': []
            }
            for t in range(0, self.max_steps):
                episode['states'].append(state)
                action, prob = self.select_action(state)
                episode['actions'].append(action)
                episode['probs'].append(prob)
                state, reward, done, info = self.env.step(action)
                episode['rewards'].append(reward)
                if done:
                    break
        return episode
    
    def update_policy(
            self,
            max_episodes: int,
            save_pth: str,
            save: bool=True,
            logging: bool=True,
            checkpoint: int=100) -> List[float]:
        '''
        Plays `max_episodes` number of episodes
        and after each episode, updates the 
        `self.policy` using the rewards generated for
        that episode and the actions taken. The policy update
        happens only after one complete episode has been
        played using the current `self.policy` and the number of
        updates done for any episode is equal to the length of the
        episode (each state, action, reward triplet corresponds to one
        update)
        Args:
            max_episodes: Total number of episodes to train
            save_pth: Save the best agent trained to this path
            checkpoint: number of episodes after which the running
                rewards are logged
            logging: True if you want to log the training rewards on screen
            save: True if you want to save the best model
        Returns:
        Return of each episode played
        '''
        training_returns = []
        running_reward = 0
        best_return = 0
        for ep_idx in range(max_episodes):
            
            episode = self.play_episode()
            self.policy.train()
            
            returns = []
            future_return = 0
            for r in episode['rewards'][::-1]:
                future_return += r
                returns.insert(0, future_return)

            training_returns.append(returns[0])
            if returns[0] >= best_return:
                best_return = returns[0]
                if save:
                    torch.save(self.policy, save_pth)
                           
            running_reward = running_reward * 0.9 +  returns[0] * 0.1
            actions = episode['actions']
            
            for i in range(len(actions)):
                self.optimizer.zero_grad()
                state = torch.tensor(episode['states'][i]).unsqueeze(0)
                probs = self.policy(state)[0]
                imp_weight = probs[actions[i]].detach().item() / episode['probs'][i]
                log_prob = torch.log(probs[actions[i]])
                loss = - returns[i] * log_prob * imp_weight
                loss.backward()
                self.optimizer.step()
            
            if logging and (ep_idx + 1) % checkpoint == 0:
                print(f"Episode: {ep_idx + 1} | score: {running_reward}")
        return training_returns


class SuttonAgentImportanceWeightedThresholded():
    def __init__(self, env_name: str):
        '''
        Args
            env_name: Gym environment to train the model on
        '''
        self.env = gym.make(env_name)
        self.observation_space_dim = self.env.observation_space._shape[0]
        self.action_space_dim = self.env.action_space.n
        self.policy = MLPPolicy(
            observation_space_dim=self.observation_space_dim,
            action_space_dim=self.action_space_dim
        )
        self.optimizer = optim.Adam(self.policy.parameters())
        self.max_steps = 100000
    
    def select_action(
            self,
            state: Iterator[float]) -> Tuple[int, float]:
        '''
        Given a `state`, samples an action using
        `self.policy`. Returns the action sampled
        '''
        state_tensor = torch.tensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        try:
            m = Categorical(action_probs)
        except:
            print(action_probs)
            m = Categorical(action_probs)
        action = m.sample()
        return action.item(), action_probs[0][action]
    
    def play_episode(self) -> Dict[str, List[float]]:
        '''
        Runs one episode of the environment
        using `self.policy`. Returns states, actions,
        and rewards observed during this episode
        '''
        self.policy.eval()
        with torch.no_grad():
            state = self.env.reset()
            episode = {
                'states': [],
                'actions': [],
                'probs' : [],
                'rewards': []
            }
            for t in range(0, self.max_steps):
                episode['states'].append(state)
                action, prob = self.select_action(state)
                episode['actions'].append(action)
                episode['probs'].append(prob)
                state, reward, done, info = self.env.step(action)
                episode['rewards'].append(reward)
                if done:
                    break
        return episode
    
    def update_policy(
            self,
            max_episodes: int,
            save_pth: str,
            save: bool=True,
            logging: bool=True,
            threshold: float = 0.1,
            checkpoint: int=100) -> List[float]:
        '''
        Plays `max_episodes` number of episodes
        and after each episode, updates the 
        `self.policy` using the rewards generated for
        that episode and the actions taken. The policy update
        happens only after one complete episode has been
        played using the current `self.policy` and the number of
        updates done for any episode is equal to the length of the
        episode (each state, action, reward triplet corresponds to one
        update)
        Args:
            max_episodes: Total number of episodes to train
            save_pth: Save the best agent trained to this path
            checkpoint: number of episodes after which the running
                rewards are logged
            logging: True if you want to log the training rewards on screen
            save: True if you want to save the best model
            threshold: importance weight minimum value allowed
        Returns:
        Return of each episode played
        '''
        training_returns = []
        running_reward = 0
        best_return = 0
        for ep_idx in range(max_episodes):
            
            episode = self.play_episode()
            self.policy.train()
            
            returns = []
            future_return = 0
            for r in episode['rewards'][::-1]:
                future_return += r
                returns.insert(0, future_return)

            training_returns.append(returns[0])
            if returns[0] >= best_return:
                best_return = returns[0]
                if save:
                    torch.save(self.policy, save_pth)
                           
            running_reward = running_reward * 0.9 +  returns[0] * 0.1
            actions = episode['actions']
            
            for i in range(len(actions)):
                self.optimizer.zero_grad()
                state = torch.tensor(episode['states'][i]).unsqueeze(0)
                probs = self.policy(state)[0]
                imp_weight = probs[actions[i]].detach().item() / episode['probs'][i]
                log_prob = torch.log(probs[actions[i]])
                loss = - returns[i] * log_prob * max(threshold, imp_weight)
                loss.backward()
                self.optimizer.step()
            
            if logging and (ep_idx + 1) % checkpoint == 0:
                print(f"Episode: {ep_idx + 1} | score: {running_reward}")
        return training_returns
            
            
        
                