"""
Implementation of Buridan's ass MOMDP from http://dx.doi.org/10.1145/1102351.1102427
"""

import gymnasium
import numpy as np

from gymnasium import spaces

class BuridansAss(gymnasium.Env):
    def __init__(self, p_stolen, n_appear, max_steps):
        super().__init__()

        self.max_time_hungry = 9
        self.p_stolen = p_stolen
        self.n_appear = n_appear
        self.max_steps = max_steps
        self.steps = 0

        self.observation_space = spaces.Tuple((
            spaces.Discrete(9), # Position
            spaces.MultiBinary(2), # Food at each pile
            spaces.Discrete(10), # Time since last meal
        ))
        self.action_space = spaces.Discrete(5)
        self.reward_space = spaces.Box(
          low=np.array([-1, -1, -1]),
          high=np.array([0, 0, 0])
        )

        self.reset()

    def reset(self, seed=None, options=None):
      super().reset(seed=seed)
      self.pos = 4  # center of 3x3 grid
      self.food = [1, 1]  # food present at both piles
      self.t_since_meal = 0
      self.steps = 0
      return self._get_obs(), {}
    
    def step(self, action):
        # Movement logic
        movement = [-3, 3, -1, 1, 0]  # up, down, left, right, stay
        next_pos = self.pos + movement[action]
        if 0 <= next_pos < 9 and not (self.pos % 3 == 0 and action == 2) and not (self.pos % 3 == 2 and action == 3):
            self.pos = next_pos

        # Hunger penalty
        reward_hunger = 0
        if self._is_food_square(self.pos) and self.food[self._food_index(self.pos)]:
            reward_hunger = 0
            self.t_since_meal = 0
            self.food[self._food_index(self.pos)] = 0
        else:
            self.t_since_meal = min(self.max_time_hungry, self.t_since_meal + 1)
            if self.t_since_meal == self.max_time_hungry:
                reward_hunger = -1

        # Food stolen penalty
        reward_stolen = 0
        for i in range(2):
            if self.food[i] and not self._is_adjacent_to_pile(i):
                if np.random.rand() < self.p_stolen:
                    self.food[i] = 0
                    reward_stolen -= 0.5

        # Walking penalty
        reward_walk = -1 if action != 4 else 0

        self.steps += 1
        if self.steps % self.n_appear == 0:
            self.food = [1, 1]

        done = self.steps == self.max_steps

        rewards = np.array([reward_hunger, reward_stolen, reward_walk], dtype=np.float32)
        return self._get_obs(), rewards, done, False, {}

    def _get_obs(self):
        return (self.pos, np.array(self.food), self.t_since_meal)

    def _food_index(self, pos):
        return 0 if pos == 0 else 1

    def _is_food_square(self, pos):
        return pos in [0, 8]

    def _is_adjacent_to_pile(self, pile_idx):
        food_pos = [0, 8][pile_idx]
        adj = {0: [1, 3, 4], 8: [4, 5, 7]}
        return self.pos in adj[food_pos]

    