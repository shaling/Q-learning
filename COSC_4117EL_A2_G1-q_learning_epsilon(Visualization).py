import numpy as np
import pygame
from gridworld_domain import GridWorld, GRID_SIZE, setup_pygame, draw_grid, SCREEN_WIDTH, SCREEN_HEIGHT
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningEpsilon:
    def __init__(self, alpha=0.1, discount_factor=0.5, epsilon=1.0):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.01
        self.actions = ["up", "down", "left", "right"]
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.action_index = {v: k for k, v in self.action_map.items()}
        self.exploration_map = np.zeros((GRID_SIZE, GRID_SIZE))
        self.delivery_cells_visited = set()

    def get_q_value(self, state, action):
        i, j = state
        action_index = self.action_index[action]
        return self.q_table[i, j, action_index]

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            random_index = np.random.randint(0, len(self.actions))
            return self.actions[random_index]
        else:
            i, j = state
            action_index = np.argmax(self.q_table[i, j])
            return self.action_map[action_index]

    def learn(self, state, action, reward, next_state, world):
        i, j = state
        next_i, next_j = next_state
        action_index = self.action_index[action]
        current_q = self.q_table[i, j, action_index]
        if next_state == world.goal:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_i, next_j])
        sample = reward + self.discount_factor * next_max_q
        new_q = current_q + self.alpha * (sample - current_q)
        self.q_table[i, j, action_index] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train(agent, episodes):
    # Initialize Pygame
    screen, clock = setup_pygame()
    episode_rewards = []
    avg_rewards = []
    delivery_counts = []
    
    for episode in range(episodes):
        random.seed(100)
        world = GridWorld()
        current_state = world.robot_pos
        total_reward = 0
        done = False
        deliveries_this_episode = 0
        visited_cells = set()
        
        # Determine if this is the last episode
        is_last_episode = episode == episodes - 1
        # Set movement delay based on episode
        move_delay = 0.5 if is_last_episode else 0.001  # Slower in last episode
        
        while not done:
            # Clear screen and draw current state
            screen.fill((255, 255, 255))
            draw_grid(world, screen)
            pygame.display.flip()
            
            # Add delay for visualization
            time.sleep(move_delay)
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return agent, episode_rewards, avg_rewards, delivery_counts

            visited_cells.add(current_state)
            action = agent.get_action(current_state)
            
            if is_last_episode:
                print(f"Episode {episode + 1}, State: {current_state}, Action: {action}")
            
            reward = world.move(action)
            next_state = world.robot_pos
            
            if reward == 18:  # Delivery reward
                deliveries_this_episode += 1
            
            total_reward += reward

            i, j = next_state
            agent.exploration_map[i, j] += 1
            agent.learn(current_state, action, reward, next_state, world)
            current_state = next_state
            done = next_state == world.goal

            if is_last_episode:
                print(f"Moved to {next_state}, Reward: {reward}, Total Reward: {total_reward}")

        delivery_counts.append(deliveries_this_episode)
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode >= 99:
            avg_r = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_r)
        else:
            avg_rewards.append(np.mean(episode_rewards))

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward={total_reward}, Avg Reward={avg_rewards[-1]:.2f}, Epsilon: {agent.epsilon:.3f}")

    pygame.quit()
    return agent, episode_rewards, avg_rewards, delivery_counts

if __name__ == "__main__":
    agent = QLearningEpsilon()
    # Train the agent
    agent, episode_rewards, avg_rewards, delivery_counts = train(agent, episodes=8000)
    
    # Plot training progress
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward', alpha=0.6)
    plt.plot(avg_rewards, label='100-Episode Average', color='red', linewidth=2)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(delivery_counts)
    plt.title('Deliveries per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Deliveries')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(agent.exploration_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit Count')
    plt.title('Exploration Heat Map')
    plt.show()