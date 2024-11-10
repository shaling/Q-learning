import numpy as np
import pygame
from gridworld_domain import GridWorld, GRID_SIZE, setup_pygame, draw_grid, SCREEN_WIDTH, SCREEN_HEIGHT
import time
import random
import matplotlib.pyplot as plt
class QLearningExploration:
    def __init__(self, alpha=0.1, dis_factor=0.9, k=1.0):
        self.alpha = alpha  # Learning rate
        self.dis_factor = dis_factor  # Discount factor
        self.k = k  # Exploration constant
        self.actions = ["up", "down", "left", "right"]
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # Q-table for state-action pairs
        self.visit_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # Visit counts for state-action pairs
        self.action_index = {v: k for k, v in self.action_map.items()}
        
        
    # exploration function with bonus rewards for less-visited states(use square root function)
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
            # Calculate the value of each action using the Q-value and exploration bonus
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
        
        # Q-learning update rule:
        # 1. Calculate the sample using the reward and the maximum Q-value of the next state
        sample = reward + self.dis_factor * next_max_q
        # 2. Update the Q-value using the learning rate and the sample
        new_q = current_q + self.alpha * (sample - current_q)
        # 3. Update the Q-value in the Q-table
        self.q_table[i, j, action_index] = new_q

## WRITE by ChatGPT
def visualize_q_values(q_table):
    """Visualize Q-values with max values highlighted in green"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # For each cell in the grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Draw cell boundaries
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, color='black'))
            
            # Get Q-values directly from q_table
            # The indices in q_table correspond to:
            # 0: up, 1: down, 2: left, 3: right
            q_up = q_table[i, j, 0]        # Move up reduces i
            q_down = q_table[i, j, 1]      # Move down increases i
            q_left = q_table[i, j, 2]      # Move left reduces j
            q_right = q_table[i, j, 3]     # Move right increases j
            
            q_values = [q_up, q_down, q_left, q_right]
            max_q = max(q_values)
            
            # Debug print
            print(f"State ({i},{j}): Up={q_up:.1f}, Down={q_down:.1f}, Left={q_left:.1f}, Right={q_right:.1f}")
            
            # Draw diagonal lines
            ax.plot([j, j+1], [i, i+1], color='gray', linestyle='--', linewidth=0.5)
            ax.plot([j, j+1], [i+1, i], color='gray', linestyle='--', linewidth=0.5)
            
            # Add Q-values with corrected positioning
            # Up value at bottom (because moving up decreases i)
            ax.text(j+0.5, i+0.25, f'{q_up:.1f}', 
                   ha='center', va='center', 
                   color='green' if q_up == max_q else 'black')
            
            # Down value at top (because moving down increases i)
            ax.text(j+0.5, i+0.75, f'{q_down:.1f}', 
                   ha='center', va='center',
                   color='green' if q_down == max_q else 'black')
            
            # Left value on left side (because moving left decreases j)
            ax.text(j+0.25, i+0.5, f'{q_left:.1f}',
                   ha='center', va='center',
                   color='green' if q_left == max_q else 'black')
            
            # Right value on right side (because moving right increases j)
            ax.text(j+0.75, i+0.5, f'{q_right:.1f}',
                   ha='center', va='center',
                   color='green' if q_right == max_q else 'black')
    
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.invert_yaxis()  # Invert y-axis to match grid coordinates
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    plt.title('Q-values by Action for Each State\n(Green indicates highest value)')
    plt.grid(True, linestyle=':')  # Add light grid for better readability
    plt.show()

def train(agent,episodes):
    episode_rewards = []
    avg_rewards = []
    screen,clock = None,None
    
    for episode in range(episodes):
        is_last_episode = episode == episodes - 1

        # Write by ChatGPT
        if is_last_episode:
            screen, clock = setup_pygame()

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

            ## WRITE by ChatGPT
            if is_last_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return agent, episode_rewards, avg_rewards
                screen.fill((255, 255, 255))
                draw_grid(world, screen)
                pygame.display.flip()
                clock.tick(1)
                time.sleep(0.1)

            
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
    
    if screen:
        pygame.quit()
    return agent, episode_rewards,avg_rewards


if __name__ == "__main__":
    # Train the agent
    k_values = [1.0, 50.0, 100.0]
    
    colors=['r','g','b']
    all_avg_rewards = [] # Store average rewards for each k value and resolve the issue of conflictting pygames
    for index_k, k in enumerate(k_values):
        agent = QLearningExploration(k=k)
        agent, episode_rewards,avg_rewards = train(agent,episodes=200)
        all_avg_rewards.append(avg_rewards)
    plt.figure(figsize=(12, 6))
    for index_k, k in enumerate(k_values):
        plt.plot(all_avg_rewards[index_k], label=f"Avg Rewards k={k}", linewidth=2, color=colors[index_k])
    
    
    plt.title("Training Progress with Different Exploration Constants (k)")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='lower right')
    plt.show()

    visualize_q_values(agent.q_table)