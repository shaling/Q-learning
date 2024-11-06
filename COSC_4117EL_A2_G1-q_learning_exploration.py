import numpy as np
import pygame
from gridworld_domain import GridWorld, GRID_SIZE, setup_pygame, draw_grid
import time
import random

class QLearningExploration:
    def __init__(self, alpha=0.1, dis_factor=0.8, k=5.0):
        self.alpha = alpha  # Learning rate
        self.dis_factor = dis_factor  # Discount factor
        self.k = k  # Exploration constant
        self.actions = ["up", "down", "left", "right"]
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # Q-table for state-action pairs
        self.visit_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # Visit counts for state-action pairs
        self.action_index = {v: k for k, v in self.action_map.items()}
        
        
        
    def get_exploration_bonus(self, state, action):
        i,j = state
        action_index = self.action_index[action]
        n= self.visit_table[i, j, action_index]+1
        return self.k / np.sqrt(n)
        
    def get_q_value(self, state, action):
        i, j = state
        action_index = self.action_index[action]
        return self.q_table[i, j, action_index]
        
    def get_action(self, state):
        i, j = state
        action_values = {}
        for action in self.actions:
            action_values[action] = self.get_q_value(state, action) + self.get_exploration_bonus(state, action)
        
        # Choose action with highest value
        return max(action_values.items(), key=lambda x: x[1])[0]
        
    def learn(self, state, action, reward, next_state, world):
        i, j = state
        next_i, next_j = next_state
        action_index = self.action_index[action]
            
        # Update visit count
        self.visit_table[i,j,action_index] += 1
        
        # Q-learning update rule
        current_q = self.q_table[i, j, action_index]
        if next_state == world.goal:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_i, next_j])
        
        sample = reward + self.dis_factor * next_max_q
        new_q = current_q + self.alpha * (sample - current_q)
        self.q_table[i, j, action_index] = new_q

def train(agent,episodes):
    episode_rewards = []
    avg_rewards = []
    
    for episode in range(episodes):
        random.seed(100)  # Reset gridworld seed
        world = GridWorld()
        total_reward = 0
        done = False
        
        while not done:
            current_state = world.robot_pos
            action = agent.get_action(current_state)
            print(f"Episode {episode + 1}, State: {current_state}, Action: {action}")
            # Take action and get reward
            reward = world.move(action)
            next_state = world.robot_pos
            
            # Learn from experience
            agent.learn(current_state, action, reward, next_state, world)
            
            total_reward += reward
            
            # Check if episode is done
            if world.robot_pos == world.goal:
                done = True
            print(f"Moved to {next_state}, Reward: {reward}, Total Reward: {total_reward}")
        episode_rewards.append(total_reward)

        if episode >= 99:  # Start averaging after 100 episodes
            avg_r = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_r)
        else:
            avg_rewards.append(np.mean(episode_rewards))

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
    
    
    return agent, episode_rewards,avg_rewards

if __name__ == "__main__":
    # Train the agent
    agent = QLearningExploration()
    agent, episode_rewards,avg_rewards = train(agent,episodes=2000)
    
    # Plot training progress
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.plot(avg_rewards, label='100-Episode Average', color='red', linewidth=2)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()