"""
Discrete water reservoir environment based on:
    A. Castelletti, F. Pianosi and M. Restelli, "Tree-based Fitted Q-iteration for Multi-Objective Markov Decision problems,"
    The 2012 International Joint Conference on Neural Networks (IJCNN),
    Brisbane, QLD, Australia, 2012, pp. 1-8, doi: 10.1109/IJCNN.2012.6252759.

"""

import gymnasium
import numpy as np
from scipy.stats import norm

from gymnasium import spaces

class Dam(gymnasium.Env):
    def __init__(
      self,
      seed=None,
      s_0=None,
      capacity=10,
      n_states=20,
      water_demand=4,
      power_demand=3,
      inflow_mean=2,
      inflow_std=1,
      episode_length = 30,
      penalize=False):
        self.rng = np.random.default_rng(seed)

        self.capacity = capacity
        self.n_states = n_states
        self.water_demand = water_demand
        self.power_demand = power_demand
        # inflow distribution
        self.inflow_mean = inflow_mean
        self.inflow_std = inflow_std
        self.penalize = penalize
        self.episode_length = episode_length

        # NOTE: this assumes that the max water release over a single timestep is the reservoir capacity (which is more than IRL)
        self.action_space = spaces.Discrete(self.capacity)

        self.observation_space = spaces.Box(
            low=0,
            high=self.n_states,
            dtype=np.int32
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
            self.state = np.int32([self.rng.integers(0, self.capacity + 1)])
        self.t = 0
        return self.state, {}
    
    def build_transition_matrix(self):
        P = np.zeros((self.n_states, self.capacity, self.n_states))

        # define inflow range for probability mass
        inflow_values = np.arange(int(self.inflow_mean - 4*self.inflow_std),
                                int(self.inflow_mean + 4*self.inflow_std) + 1)

        # compute probabilities of each discrete inflow
        inflow_probs = norm.pdf(inflow_values, self.inflow_mean, self.inflow_std)
        inflow_probs /= inflow_probs.sum()  # normalize

        for s in range(self.n_states):
            for a in range(self.capacity):
                for inflow, p_inflow in zip(inflow_values, inflow_probs):
                    s_next = int(np.clip(s - a + inflow, 0, self.n_states - 1))
                    P[s, a, s_next] += p_inflow

        # ensure probabilities sum to 1 across next states
        P /= P.sum(axis=2, keepdims=True)
        return P
    def build_reward_matrix(self):
        R = np.zeros((self.n_states, self.capacity, self.n_states, 2))  # [r1, r2]
        for s in range(self.n_states):
            for a in range(self.capacity):
                for s_next in range(self.n_states):
                    #TODO: add penalty
                    supply_error = np.clip(s_next - self.water_demand, None, 0)
                    r1 = supply_error

                    deficit = max(0, self.power_demand - a)
                    r2 = -deficit

                    R[s, a, s_next, 0] = r1
                    R[s, a, s_next, 1] = r2
        return R
    def step(self, action):
        self.t += 1
        # ensure that at least the excess amount of water is released
        actionLB = np.clip(self.state - self.capacity, 0, None)
        actionUB = self.capacity

        # Penalty proportional to the violation
        bounded_action = np.clip(action, actionLB, actionUB)
        penalty = -self.penalize * np.abs(bounded_action - action)
        action = bounded_action

        # compute dam inflow
        inflow = int(round(self.rng.normal(self.inflow_mean, self.inflow_std)))
        n_state = np.clip(self.state - action + inflow, 0, None).astype(np.int64)

        """# Flooding objective
        overflow = np.clip(n_state - self.capacity, 0, None)[0]
        r0 = -overflow + penalty"""

        # Deficit in water supply w.r.t. demand
        supply_error = np.clip(n_state - self.water_demand, None, 0)[0]
        r1 = supply_error + penalty

        # deficit in hydro-electric power supply
        #deficit = np.clip(self.power_demand - action, 0, None)[0]
        deficit = max(0, self.power_demand - action)
        r2 = -deficit + penalty

        """# Flood risk downstream
        flood_risk = np.clip(action - self.flood_threshold, 0, None)[0]
        r3 = -flood_risk + penalty"""

        reward = np.array([r1, r2], dtype=np.float32).flatten()

        self.state = n_state

        return n_state, reward, self.t == self.episode_length, False, {}