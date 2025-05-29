"""
Discrete water reservoir environment based on:
    A. Castelletti, F. Pianosi and M. Restelli, "Tree-based Fitted Q-iteration for Multi-Objective Markov Decision problems,"
    The 2012 International Joint Conference on Neural Networks (IJCNN),
    Brisbane, QLD, Australia, 2012, pp. 1-8, doi: 10.1109/IJCNN.2012.6252759.

"""

import gymnasium
import numpy as np

from gymnasium import spaces

class Dam(gymnasium.Env):
    def __init__(self, seed=None, s_0=None, penalize=True):
        self.rng = np.random.default_rng(seed)

        self.capacity = 25 # Max capacity of reservoir
        self.demand = 7 # Water demand
        # inflow distribution
        self.inflow_mean = 5.0
        self.inflow_std = 2.0
        self.penalize=penalize

        #self.n_capacity_levels = self.capacity + 1

        self.action_space = spaces.Box(
            low=0,
            high=np.inf,
            dtype=np.int64
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            dtype=np.int64
        )
        self.reward_space = spaces.Box(
          low=np.array([-np.inf, -np.inf]),
          high=np.zeros(2),
          dtype=np.float32
        )

        self.s_0 = s_0
        self.state = s_0
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.s_0 is not None:
            self.state = self.s_0
        else:
            self.state = np.int64([self.rng.integers(0, self.capacity + 1)])
        self.t = 0
        return self.state, {}

    def step(self, action):
        self.t += 1
        # bound the action
        actionLB = np.clip(self.state - self.capacity, 0, None)
        actionUB = self.state

        # Penalty proportional to the violation
        bounded_action = np.clip(action, actionLB, actionUB)
        penalty = -self.penalize * np.abs(bounded_action - action)
        action = bounded_action

        # compute dam inflow
        inflow = int(round(self.rng.normal(self.inflow_mean, self.inflow_std)))
        n_state = np.clip(self.state - action + inflow, 0, None).astype(np.int64)

        # Flooding objective
        overflow = np.clip(n_state - self.capacity, 0, None)[0]
        r0 = -overflow + penalty
        #n_state = min(n_state, self.capacity) # TODO: decide whether to clip or leave overflow

        # Deficit in water supply w.r.t. demand
        supply_error = -np.clip(self.demand - action, 0, None)[0]
        r1 = supply_error + penalty

        reward = np.array([r0, r1], dtype=np.float32).flatten()

        self.state = n_state

        return n_state, reward, self.t > 50, False, {}