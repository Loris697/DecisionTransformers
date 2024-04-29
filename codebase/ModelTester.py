import torch
import torch.nn as nn

import gymnasium as gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Callable
import imageio

import numpy as np

class ModelTester:
    def __init__(self, 
                 env_name: str,
                 model: nn.Module,
                 device: str = "cuda",
                 seq_len: int = 32,
                 render_mode: str = "human",
                 actionCheck: Callable = None,
                 max_reward: float = 800.,
                ):
        """Initialize the tester with an environment and a model."""
        self.env = DummyVecEnv([lambda: gym.make(env_name, render_mode=render_mode)])
        self.model = model.to(device=device)
        self.device = device
        self.seq_len = seq_len
        self.actionCheck = actionCheck
        self.max_reward = max_reward
        self.render_mode = render_mode

        #defining the inputs
        self.rewards = torch.zeros(1, seq_len, 1)
        self.observations = torch.zeros(1, seq_len, self.env.observation_space.shape[2], self.env.observation_space.shape[0], self.env.observation_space.shape[1])
        self.actions = torch.zeros(1, seq_len, self.env.action_space.shape[0])
    
    def run_episode(self, 
                    render: bool = False,
                    starting_reward: float = 1.,
                    file_name: str = "episode_output",
                    starting_action: gym.spaces.Space = None
                   ):
        """Run one episode to test the model."""
        state = self.env.reset()
        done = False
        total_reward = 0
        step = 0 
        if starting_action is None:
            starting_action = self.env.action_space.sample()

        #init the sequence
        self.reset_sequence()
        
        self.rewards[0][step] = torch.tensor(starting_reward)
        self.actions[0][step] = torch.tensor(starting_action)
        self.observations[0][step] = torch.tensor(np.array(state)/ 255. ).squeeze(0).permute(2, 0, 1)

        frames = []  # to store frames for gif

        while not done:
            if self.render_mode == "human" and render:
                self.env.render()
            if self.render_mode == "rgb_array" and render:
                frame = self.env.render()
                frames.append(frame)

            action = self.model({
                "rewards" : self.rewards.to(device=self.device, dtype=torch.float32),
                "observations" : self.observations.to(device=self.device, dtype=torch.float32),
                "actions" : self.actions.to(device=self.device, dtype=torch.float32),
            })
            next_action = np.array(action[0][step].cpu().detach())
            if self.actionCheck is not None:
                next_action = self.actionCheck(next_action)
            
            state, reward, done, info = self.env.step([next_action])
            
            next_reward = self.rewards[0][0].cpu().item() * self.max_reward - reward

            # updating the sequence of rewards, observations, actions
            if step < self.seq_len - 1:
                step += 1
                self.rewards[0][step] = torch.tensor(next_reward/self.max_reward)
                self.actions[0][step] = torch.tensor(next_action)
                self.observations[0][step] = torch.tensor(np.array(state)/ 255. ).squeeze(0).permute(2, 0, 1)
            else:
                self.rewards = torch.cat([self.rewards[:,1:], torch.tensor(next_reward/self.max_reward).reshape(1, 1, -1)], dim=1)
                self.observations = torch.cat([self.observations[:,1:], torch.tensor(state/255.).permute(0, 3, 1, 2).unsqueeze(0)], dim=1) 
                self.actions = torch.cat([self.actions[:,1:], torch.tensor(next_action).reshape(1, 1, -1)], dim=1) 
            
            total_reward += reward

        if self.render_mode == "rgb_array" and render:
            # Create a GIF from the captured frames
            gif_path = file_name + '.gif'
            imageio.mimsave(gif_path, frames, fps=30)  # Adjust fps according to your needs
            #print(f"Saved episode gif to {gif_path}")
            
        return total_reward
    
    def test_model(self, episodes=100, 
                   render=False, 
                   starting_rewards = [1.],
                   folder = "output_folder"
                  ):
        """Test the model over a number of episodes and average the rewards."""

        average_reward_starting = {}

        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for starting_reward in starting_rewards:
            total_rewards = [self.run_episode(render=render,
                                              starting_reward= starting_reward,
                                              file_name = folder + "/" + "starting_rewards"+ str(starting_reward) + "_" +str(iteration)
                                             ) for iteration, _ in enumerate(range(episodes))]
            average_reward = sum(total_rewards) / episodes
            average_reward_starting[str(starting_reward)] = average_reward
        return average_reward_starting

    def close_env(self):
        """Close the Gym environment."""
        self.env.close()

    def reset_sequence(self):
        """Deleting the sequences."""
        self.rewards = torch.zeros(1, self.seq_len, 1)
        self.observations = torch.zeros(1, self.seq_len, self.env.observation_space.shape[2], self.env.observation_space.shape[0], self.env.observation_space.shape[1])
        self.actions = torch.zeros(1, self.seq_len, self.env.action_space.shape[0])