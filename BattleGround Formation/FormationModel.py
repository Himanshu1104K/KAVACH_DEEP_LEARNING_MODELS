import gym
import numpy as np
import tensorflow as tf
import pygame
from stable_baselines3 import PPO
from gym import spaces

# Constants for visualization
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
GRID_SIZE = 10
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Soldiers
RED = (255, 0, 0)  # Enemies
YELLOW = (255, 255, 0)  # Injured Soldiers


class SoldierBattleEnv(gym.Env):
    def __init__(self):
        super(SoldierBattleEnv, self).__init__()

        # Define the observation space (health, position, threats)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )

        # Define the action space (formation changes, rescue moves)
        self.action_space = spaces.Discrete(5)

        # Initialize Pygame for Visualization
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Soldier BattleField Simulation")
        self.clock = pygame.time.Clock()

        # Soldier and Enemy Positions
        self.soldier_positions = np.random.randint(0, GRID_SIZE, (5, 2))
        self.enemy_positions = np.random.randint(0, GRID_SIZE, (3, 2))

    def reset(self):
        self.soldier_positions = np.random.randint(0, GRID_SIZE, (5, 2))
        self.enemy_positions = np.random.randint(0, GRID_SIZE, (3, 2))
        self.state = np.random.rand(10)

        return self.state

    def step(self, action):
        self.update_soldier_position(action)
        reward = self.compute_reward(action)
        done = False
        self.render()
        return np.random.rand(10), reward, done, {}

    def compute_reward(self, action):
        return np.random.rand() * 10 - 5

    def update_soldier_position(self, action):
        # Move soldiers based on action
        for i in range(len(self.soldier_positions)):
            if action == 0:  # Defensive formation
                self.soldier_positions[i][0] = max(0, self.soldier_positions[i][0] - 1)
            elif action == 1:  # Aggressive advance
                self.soldier_positions[i][0] = min(
                    GRID_SIZE - 1, self.soldier_positions[i][0] + 1
                )
            elif action == 2:  # Rescue injured
                self.soldier_positions[i][1] = max(0, self.soldier_positions[i][1] - 1)
            elif action == 3:  # Spread formation
                self.soldier_positions[i][1] = min(
                    GRID_SIZE - 1, self.soldier_positions[i][1] + 1
                )

    def render(self):
        # Render the battlefield using Pygame
        self.screen.fill(WHITE)

        # Draw Grid
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, SCREEN_HEIGHT))

        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (SCREEN_WIDTH, y))

        # Draw soldiers (Green)
        for pos in self.soldier_positions:
            pygame.draw.circle(
                self.screen,
                GREEN,
                (
                    pos[0] * CELL_SIZE + CELL_SIZE // 2,
                    pos[1] * CELL_SIZE + CELL_SIZE // 2,
                ),
                10,
            )

        # Draw enemies (Red)
        for pos in self.enemy_positions:
            pygame.draw.circle(
                self.screen,
                RED,
                (
                    pos[0] * CELL_SIZE + CELL_SIZE // 2,
                    pos[1] * CELL_SIZE + CELL_SIZE // 2,
                ),
                10,
            )

        pygame.display.flip()
        self.clock.tick(2)  # Limit frame rate to 2 FPS for better visibility


env = SoldierBattleEnv()
model = PPO('MlpPolicy',env,verbose=1)
model.learn(total_timesteps= 1000)

# Save and Test the Model
model.save("Soldier_formation_model")

obs = env.rest()
for _ in range(50):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action,_states = model.predict(obs)
    obs, rewards, done, _ = env.step(action)