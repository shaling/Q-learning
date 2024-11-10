import numpy as np
import pygame
from gridworld_domain import GridWorld, GRID_SIZE, setup_pygame, draw_grid, SCREEN_WIDTH, SCREEN_HEIGHT
import random
import matplotlib.pyplot as plt

class QLearningEpsilon:
    def __init__(self, alpha=0.1, discount_factor=0.9, epsilon=1.0):
        self.alpha = alpha # Learning rate
        self.discount_factor = discount_factor 
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = 0.9995 # Decay rate
        self.min_epsilon = 0.001 # Minimum exploration rate
        self.actions = ["up", "down", "left", "right"] # Possible actions
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"} # Map action index to action
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4)) # Q-table
        self.action_index = {v: k for k, v in self.action_map.items()} # Map action to action index
        self.exploration_map = np.zeros((GRID_SIZE, GRID_SIZE)) # Track how many times each cell is visited

    # Get the Q-value for a given state-action pair
    def get_q_value(self, state, action):
        i, j = state
        action_index = self.action_index[action]
        return self.q_table[i, j, action_index]

    # Get the best action for a given state
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

        # Q-learning update rule:
        # 1. Calculate the sample using the reward and the maximum Q-value of the next state
        sample = reward + self.discount_factor * next_max_q
        # 2. Update the Q-value using the learning rate and the sample
        new_q = current_q + self.alpha * (sample - current_q)
        # 3. Update the Q-value in the Q-table
        self.q_table[i, j, action_index] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train(agent, episodes):
    # Initialize Pygame
    screen, clock = None, None
    episode_rewards = []
    avg_rewards = []
    delivery_counts = []
    
    for episode in range(episodes):
        is_last_episode = episode == episodes - 1
        if is_last_episode:
            screen, clock = setup_pygame()
        random.seed(100)
        world = GridWorld()
        current_state = world.robot_pos
        total_reward = 0
        done = False
        deliveries_this_episode = 0
        visited_cells = set()
        
        while not done:
            if is_last_episode and screen is not None:
                # print(agent.q_table)
                screen.fill((255, 255, 255))
                draw_grid(world, screen)
                pygame.display.flip()
                clock.tick(1)
            
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return agent, episode_rewards, avg_rewards, delivery_counts

            visited_cells.add(current_state)
            action = agent.get_action(current_state)
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

            print(f"Moved to {next_state}, Reward: {reward}, Total Reward: {total_reward}")
        print(f"Episode {episode + 1}: Deliveries = {deliveries_this_episode}, Total Reward = {total_reward},, Epsilon: {agent.epsilon:.3f}")
        delivery_counts.append(deliveries_this_episode)
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode >= 99:
            avg_r = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_r)
        else:
            avg_rewards.append(np.mean(episode_rewards))

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward={total_reward}, Avg Reward={avg_rewards[-1]:.2f}")

    if screen:
        pygame.quit()
    return agent, episode_rewards, avg_rewards, delivery_counts

# write by ChatGPT
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


if __name__ == "__main__":
    agent = QLearningEpsilon()
    # Train the agent
    agent, episode_rewards, avg_rewards, delivery_counts = train(agent, episodes=12000)
    
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
    plt.pause(0.1)  

    plt.figure(figsize=(8, 8))
    plt.imshow(agent.exploration_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit Count')
    plt.title('Exploration Heat Map')
    plt.show()

    visualize_q_values(agent.q_table)
    plt.pause(1)