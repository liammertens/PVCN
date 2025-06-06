import os
from abc import ABC
from dataclasses import dataclass
import random
from typing import List, Optional, Type, Union
from collections import defaultdict

import heapq
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import copy


from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import get_non_dominated_inds
from morl_baselines.common.performance_indicators import hypervolume
from gymnasium import spaces
import csv



@dataclass
class Transition:
    """Transition dataclass."""
    observation: np.ndarray
    action: Union[float, int]
    reward: np.ndarray
    return_: np.ndarray
    observed_return: np.ndarray
    horizon: int
    next_observation: np.ndarray
    prob: float
    terminal: bool

class EpisodeStateData:
    """
    Contains data for retrieving a transition:
        - saved under state_to_eps[s]
        - Find episode: ep_to_transitions[id]
        - use idx to get the transition from s
    """
    def __init__(self, id, idx):
        self.id = id
        self.idx = idx
    def __repr__(self):
        return "EpisodeStateData(%s, %s)" % (self.id, self.idx)
    def __eq__(self, value):
        return isinstance(value, EpisodeStateData) and value.id == self.id and value.idx == self.idx
    def __hash__(self):
        return hash(self.__repr__())

def crowding_distance(points):
    """
    Compute the crowding distance of a set of points.
    The max value (for points in extremas) is num_objectives
    """
    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding

class ValueNet(nn.Module):
    """Value network V(s, R') (critic)"""
    def __init__(self, state_dim: int, reward_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_emb = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(reward_dim+1, hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, reward_dim),
        )
    def forward(self, state, desired_return, horizon):
        s = self.state_emb(state)
        c = th.cat((desired_return, horizon), dim=1)
        r = self.c_emb(c)
        pred = self.fc(s*r)
        return pred

class PolicyNet(nn.Module):
    """
    Policy network pi(a|s, R') (actor):
        Only supports discrete actions
    """
    def __init__(self, state_dim: int, reward_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_emb = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(reward_dim+1, hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, state, desired_return, horizon):
        s = self.state_emb(state.float())
        c = th.cat((desired_return, horizon), dim=1)
        r = self.c_emb(c)
        pred = self.fc(s*r)
        return pred

class PopNet(nn.Module):
    """
    POP following MLP.
    Returns the correct desired value vector for a policy that is being followed.
    It is part of the actor as the suggested desired value vector is used by it.
    Code from: https://github.com/rradules/POP-following
    """

    def __init__(self, d_in, d_out):
        super(PopNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.Tanh(),
            nn.Linear(64, d_out),
        )
    def forward(self, x):
        return self.layers(x)

class MO_AWR(MOAgent, MOPolicy):
    def __init__(
            self,
            env: gym.Env,
            policy_lr: float = 1e-4,
            value_lr: float = 1e-4,
            popf_lr: float = 1e-4,
            gamma: float = 1,
            td_lambda: float = 0.95,
            beta = 1,
            alpha = .01,
            batch_size: int = 4,
            max_buffer_size: int = 1024,
            use_popf: bool = True,
            use_is_weighting=True,
            min_exploration_stdev = 0,
            device: Union[th.device, str] = "auto",
            log: bool = False,
            wandb_entity: Optional[str] = None,
            project_name: str = "Thesis",
            experiment_name: str = "MO-AWR",
            seed: Optional[int] = None):
        """
        Args:
            env: Gym environment.
            learning_rate: Learning rate
            gamma: Discount factor
            td_lambda: Lambda parameter for TD update
            cd_threshold: Determines min. crowding distance for using simple L2-distance
            batch_size: Maximum batch size for learning.
            max_buffer_size: Maximum size of ER buffer (= amount of trajectories)
            us_popf: Whether to use POPF network
            use_is_weighting: Whether to use importance sampling ratio for weighting advantages
            min_explorations_stdev: stdev to use when only a single return is in the PF.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        self.experience_replay: List[Transition] = []
        self.state_to_eps = defaultdict(set) # maps a state-action pair to episode ids which contain that pair
        self.ep_to_transitions = defaultdict(list) # maps an episode id to its corresponding transitions

        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.popf_lr = popf_lr
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.use_popf = use_popf
        self.use_is_weighting = use_is_weighting
        self.min_expl_stdev = min_exploration_stdev
        self.beta = beta
        self.alpha = alpha

        self.pf_points = [] # tuples (return, horizon)

        # initialize networks
        th.autograd.set_detect_anomaly(True, )
        self.value_net = ValueNet(self.observation_dim, self.reward_dim)
        self.policy = PolicyNet(self.observation_dim, self.reward_dim, self.action_dim)

        self.val_opt = th.optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.val_sched = th.optim.lr_scheduler.CosineAnnealingLR(self.val_opt, 50000, 1e-5)
        self.policy_opt = th.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        #self.policy_sched = th.optim.lr_scheduler.CosineAnnealingLR(self.policy_opt, 50000, 1e-5)
        if use_popf:
            self.popf = PopNet(self.reward_dim + 1 + self.observation_dim*2, self.reward_dim)
            self.popf_opt = th.optim.Adam(self.popf.parameters(), lr=self.popf_lr)
            self.popf_sched = th.optim.lr_scheduler.CosineAnnealingLR(self.popf_opt, 50000, 7e-5)

        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "value_lr": self.value_lr,
            "policy_lr": self.policy_lr,
            "popf_lr": self.popf_lr,
            "TD_lambda": self.td_lambda,
            "buffer_size": self.max_buffer_size,
            "use_popf": self.use_popf,
            "seed": self.seed,
            "use_IS_weights": self.use_is_weighting,
        }

    # code from https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/multi_policy/pcn/pcn.py
    def _add_episode(self, transitions: List[Transition], max_size: int, step: int, fill_buffer=False) -> None:
        # compute return and add to dictionaries
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].horizon += transitions[i+1].horizon
            # First compute MC return for use as value function input
            transitions[i].observed_return += self.gamma * transitions[i + 1].observed_return
            if (not fill_buffer) and self.td_lambda < 1:
                transitions[i].return_ = transitions[i].reward + self.gamma * ((1-self.td_lambda) *
                                                                               self.value_net(th.FloatTensor(transitions[i+1].observation).unsqueeze(0).to(self.device),
                                                                                              th.FloatTensor(transitions[i+1].observed_return).unsqueeze(0).to(self.device),
                                                                                              th.FloatTensor((np.array(transitions[i+1].horizon)).reshape(1,1)).to(self.device)).detach().numpy()[0] +
                                                                               self.td_lambda*transitions[i + 1].return_)
            else:
                transitions[i].return_ += self.gamma * transitions[i + 1].return_

            if isinstance(self.env.observation_space, spaces.Box):
                state = transitions[i+1].observation.tobytes()
            elif isinstance(self.env.observation_space, spaces.Discrete):
                state = transitions[i+1].observation
            action = transitions[i+1].action
            # add episode id (= step) and index of this transition to the dict
            ep_data = EpisodeStateData(step, i+1)
            key = (state, action)
            self.state_to_eps[key].add(ep_data)
        if len(transitions) > 1:
            # update horizon from initial state!
            transitions[0].horizon = transitions[1].horizon + 1
        # also add first transition of the list
        if isinstance(self.env.observation_space, spaces.Box):
            state = transitions[0].observation.tobytes()
        elif isinstance(self.env.observation_space, spaces.Discrete):
            state = transitions[0].observation
        # NOTE: currently only discrete actions are supported...
        action = transitions[0].action
        self.state_to_eps[(state, action)].add(EpisodeStateData(step, 0))
        # Add transitions to dict
        self.ep_to_transitions[step] = transitions

        # pop smallest (= lowest dominating score) episode of heap if full, add new episode
        # heap is sorted by neg. L2 distance to closest ND point, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        if len(self.experience_replay) == max_size:
            old = heapq.heappushpop(self.experience_replay, (1, step, transitions))
            keys_to_skip = set() # avoid deleting twice from dict (unnecessary and could lead to KeyError)
            for idx, t in enumerate(old[2]):
                # remove all entries in state_to_eps
                old_data = EpisodeStateData(old[1], idx)
                if isinstance(self.env.observation_space, spaces.Box):
                    s = t.observation.tobytes()
                elif isinstance(self.env.observation_space, spaces.Discrete):
                    s = t.observation
                a = t.action
                key = (s, a)
                if not key in keys_to_skip:
                    self.state_to_eps[key].remove(old_data)
                    if len(self.state_to_eps[key]) == 0:
                        keys_to_skip.add(key)
                        # remove key from dict if empty list
                        del self.state_to_eps[key]
            del self.ep_to_transitions[old_data.id]
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _update_er_distances(self, threshold=.2, keep_duplicates=False):
        """See Section 4.4 of https://arxiv.org/pdf/2204.05036.pdf for details."""
        returns = np.array([e[2][0].return_ for e in self.experience_replay])
        # crowding distance of each point, check ones that are too close together
        distances = crowding_distance(returns)
        sma = np.argwhere(distances <= threshold).flatten()

        non_dominated_i = get_non_dominated_inds(returns)
        non_dominated = returns[non_dominated_i]
        # we will compute distance of each point with each non-dominated point,
        # duplicate each point with number of non_dominated to compute respective distance
        returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(non_dominated), 1))
        # distance to closest non_dominated point
        l2 = np.min(np.linalg.norm(returns_exp - non_dominated, axis=-1), axis=-1) * -1
        # all points that are too close together (crowding distance < threshold) get a penalty
        non_dominated_i = np.nonzero(non_dominated_i)[0]
        if not keep_duplicates:
            _, unique_i = np.unique(non_dominated, axis=0, return_index=True)
            unique_i = non_dominated_i[unique_i]
            duplicates = np.ones(len(l2), dtype=bool)
            duplicates[unique_i] = False
            l2[duplicates] -= 1e-5
        l2[sma] *= 2
        # update all distances in heap
        for i in range(len(l2)):
            self.experience_replay[i] = (l2[i], self.experience_replay[i][1], self.experience_replay[i][2])
        heapq.heapify(self.experience_replay)

    def _add_to_pf(self, returns, keep_oldest=False):
        """
        Adds a set of points to the PF:
            1. Keep non dominated points
            2. Sort based on crowding distance
            3. keep ND points with highest CD
        Args:
            returns: list of tuples (return, horizon)
        """
        for r in returns:
            self.pf_points.append(r)
        nd_inds = get_non_dominated_inds(np.array([ret[0] for ret in self.pf_points]))
        # Only keep ND points
        self.pf_points = np.array(self.pf_points, dtype=object)[nd_inds]
        # avoid unncessary computations + shape mismatch when computing CD for single point
        if self.pf_points.shape[0] > 1:
          distances = crowding_distance(np.array([nd[0] for nd in self.pf_points], dtype=object))
          sorted_inds = np.argsort(distances) # NOTE: stable arg not supported in this np version...
          # only keep best num_pf_points
          sorted_inds = sorted_inds[-self.num_pf_points:]
          self.pf_points = self.pf_points[sorted_inds].tolist()
        else:
          self.pf_points = self.pf_points.tolist()

    def _choose_commands(self):
        """
        Pick a desired return from the PF and increase it in one objective based on the mean/stdev of the PF returns.
        If only a single return is present, use a user-defined stdev
        """
        # pick random objective
        r_i = self.np_random.integers(0, self.reward_dim)
        # pick random returns
        ret_idx = self.np_random.integers(0, len(self.pf_points))
        stdev = np.std(np.array([p[0] for p in self.pf_points]), axis=0)[r_i]
        desired_return = np.array(self.pf_points[ret_idx], dtype=object)
        # make a deep copy as we are working with np array of objects!
        c = copy.deepcopy(desired_return)
        # ensure some exploration is still done when only one sample is in PF
        if self.min_expl_stdev != 0 and stdev < self.min_expl_stdev:
            stdev = self.min_expl_stdev
        c[0][r_i] += self.np_random.uniform(0, stdev)
        desired_horizon = c[1]
        # decrease horizon in attempt to improve
        # do not limit to be positive so that model learns this
        desired_horizon -= 2

        return np.float32(c[0]), np.float32(desired_horizon)

    def update_value_function(self, states, mean_rs, returns, rewards, horizons):
        mean_rs = np.array(mean_rs)
        states = th.FloatTensor(np.array(states)).to(self.device)
        returns = np.array(returns)
        horizons = th.FloatTensor(np.array(horizons)).unsqueeze(1).to(self.device)
        rewards = np.array(rewards)

        self.value_net.train()
        self.val_opt.zero_grad()
        preds = self.value_net(states, th.FloatTensor(returns).to(self.device), horizons)

        # subtract observed reward and add mean reward
        targets =  th.FloatTensor(returns - rewards + mean_rs)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        self.val_opt.step()
        self.val_sched.step()
        self.value_net.eval()
        return loss, preds

    def update_popf(self, N, states, actions, next_states, next_Rs):
        states = th.FloatTensor(np.array(states))
        actions = th.FloatTensor(np.array(actions)).unsqueeze(1)
        next_states = th.FloatTensor(np.array(next_states))
        N = th.tensor(np.float32(N))
        x = th.cat((N, states, actions, next_states), dim=1).float().to(self.device)
        targets = th.FloatTensor(np.array(next_Rs))
        self.popf.train()
        self.popf_opt.zero_grad()

        preds = self.popf(x)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        self.popf_opt.step()
        self.popf_sched.step()
        self.popf.eval()
        return loss, preds

    def _dominates(self, a, b):
        return np.all(a >= b) and np.any(a > b)
    def compute_advantages(self, returns, V_s):
        """
        Computed advantage weights for policy loss:
            - If the return dominates the desired return, use L2-norm
            - Else, use the neg. L2-norm
        """
        weights = []
        pairs = tuple(zip(returns, V_s))
        res = [(i, self._dominates(R, V)) for i, (R, V) in enumerate(pairs)]
        for i, dominates in res:
            if dominates:
                A = np.linalg.norm(returns[i]-V_s[i])
            else:
                A = -np.linalg.norm(returns[i]-V_s[i])
            weights.append(A)
        weights = np.array(weights)
        # normalize weights
        weights = (weights - weights.mean()) / (weights.std() + 1e-8)
        return th.tensor(weights)
    def compute_policy_loss(self, preds, actions, returns, expected_returns, probs):
        """
        Loss is computed using advantage estimates. Advantage weights are weighted by IS ratio.
        An entropy term is added to ensure exploration
        """
        if self.use_is_weighting:
          is_ratios = (preds[th.arange(self.batch_size), actions] / th.FloatTensor(np.array(probs))).detach()
        else:
          is_ratios = th.ones(self.batch_size)
        advantages = self.compute_advantages(returns, expected_returns)
        entropy = -th.sum(preds * th.log(preds + 1e-8), dim=1).mean()
        loss = -th.mean(th.log(preds[th.arange(self.batch_size), actions] + 1e-8)*is_ratios*th.exp(advantages/self.beta)) - self.alpha*entropy
        return loss
    def update_policy(self, states, actions, returns, horizons, expected_returns, probs):
        states = th.FloatTensor(np.array(states)).to(self.device)
        horizons = th.FloatTensor(np.array(horizons)).unsqueeze(1).to(self.device)

        self.policy_opt.zero_grad()
        preds = self.policy(states, th.FloatTensor(expected_returns).to(self.device), horizons)
        loss = self.compute_policy_loss(preds, actions, returns, expected_returns, probs)
        loss.backward()
        self.policy_opt.step()
        #self.policy_sched.step()
        return loss, preds

    def _act(self, state, desired_return, desired_horizon, eval_mode=False):
        pred = self.policy(
            th.unsqueeze(th.FloatTensor(state), 0).to(self.device),
            th.unsqueeze(th.FloatTensor(desired_return), 0).to(self.device),
            th.unsqueeze(th.FloatTensor([desired_horizon]), 0).to(self.device),
        )
        probs = pred.detach().cpu().numpy()[0]

        if eval_mode:
            action = np.argmax(probs)
        else:
            action = self.np_random.choice(np.arange(len(probs)), p=probs) # exponentiate to get proper prob.
        return action, probs[action]

    def _run_episode(self, desired_return, desired_horizon, eval_mode=False):
        transitions = []
        obs, _ = self.env.reset()
        done = False
        N = desired_return
        while not done:
            action, prob = self._act(obs, desired_return, desired_horizon, eval_mode)
            n_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            transitions.append(Transition(obs, action, np.float32(reward).copy(), np.float32(reward).copy(), np.float32(reward).copy(), 1, n_obs, prob, terminated))

            desired_horizon = np.float32(max(desired_horizon - 1, 1.0)) # avoid neg. horizon
            if self.use_popf:
                N = (N-reward)/self.gamma
                x = np.concatenate((N, obs, [action], n_obs))
                desired_return = self.popf(
                    th.tensor(x).float().to(self.device)
                ).detach().numpy()
                obs = n_obs
            else:
                obs = n_obs
                desired_return = (desired_return - reward) / self.gamma   
        return transitions

    def eval(self, obs, w=None):
        return self._act(obs, self.desired_return, eval_mode=True)

    def evaluate_pf(self, num_iterations=1):
        """
        evaluates PF points for a number of iterations and returns their mean returns.

        Returns: tuple(return, horizon)
        """
        eval_returns = []
        for i in range(len(self.pf_points)):
            r = self.pf_points[i][0]
            h = self.pf_points[i][1]
            mean = [np.zeros(self.reward_dim), 0]
            for _ in range(num_iterations):
                transitions = self._run_episode(r, h, True)
                for i in reversed(range(len(transitions) - 1)):
                    transitions[i].return_ += self.gamma * transitions[i + 1].return_
                mean[0] = mean[0] + transitions[0].return_
                mean[1] = mean[1] + len(transitions)
            mean[0] = mean[0] / num_iterations
            mean[1] = mean[1] / num_iterations
            eval_returns.append(mean)
        return eval_returns
    
    def prune_pf(self, eval_pf, threshold):
        """
        Prunes the set of PF points using their corrseponding evaluations.
        If the difference between a point and its evaluation < threshold, keep it in the PF.
        If multiple points result in the same evaluation, keep the one that resembles the evaluation most

        returns: evaluations of kept pf points
        """
        new_pf = []
        evaluations = []
        diffs = []
        for i in range(len(eval_pf)):
            diff = np.absolute(eval_pf[i][0] - self.pf_points[i][0])
            if np.all(diff < threshold):
                new_pf.append(self.pf_points[i])
                evaluations.append(eval_pf[i])
                diffs.append(np.linalg.norm(diff, ord=1))
                
        unique_dict = {}
        for i in range(len(evaluations)):
            if i not in unique_dict or diff[i] < unique_dict[i][0]:
                unique_dict[i] = (new_pf[i], evaluations[i])
        new_pf = [p[0] for p in unique_dict.values()]
        evaluations = [p[1] for p in unique_dict.values()]
        self.pf_points = new_pf
        return evaluations

    def save(self, checkpoint, savedir: str = "weights"):
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        th.save(self.value_net, f"{savedir}/value_{checkpoint}.pt")
        th.save(self.value_net, f"{savedir}/policy_{checkpoint}.pt")
        if self.use_popf:
            th.save(self.value_net, f"{savedir}/popf_{checkpoint}.pt")
    
    def get_batch(self):
        """
        Get a batch of (s, a, r, s', R, r^, H)
        s and a are the same for each computation of r^:
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        returns = []
        mean_Rs = []
        horizons = []
        probs = []
        mc_returns = []
        is_init = [] # indicates whether state was initial
        mean_R = 0

        key = random.choice(list(self.state_to_eps.keys()))
        eps_data = np.array(list(self.state_to_eps[key]), dtype=object)
        eps_data = self.np_random.choice(eps_data, self.batch_size)
        for e in eps_data:
            t: Transition = self.ep_to_transitions[e.id][e.idx]
            states.append(t.observation)
            actions.append(t.action)
            rewards.append(np.float32(t.reward))
            next_states.append(t.next_observation)
            returns.append(np.float32(t.return_))
            mc_returns.append(np.float32(t.observed_return))
            mean_R += np.float32(t.reward)
            horizons.append(np.float32(t.horizon))
            probs.append(np.float32(t.prob))
            if e.idx == 0:
                is_init.append(1)
            else:
                is_init.append(0)

        mean_R = mean_R/self.batch_size
        for _ in range(self.batch_size):
            mean_Rs.append(mean_R)       

        return states, actions, rewards, next_states, returns, mc_returns, mean_Rs, horizons, probs, is_init

    

    def update(self):
        states = [] # shape = (num_episodes, batch_size, obs_dim)
        actions = []
        rewards = []
        next_states = []
        returns = [] # TD returns
        mc_returns = []
        mean_Rs = [] # batch means over immediate rewards
        horizons = []
        probs = []
        is_init = []

        expected_Rs = []

        max_steps = max(self.num_policy_steps, self.num_value_steps, self.num_popf_steps)
        current_steps = 0
        while current_steps < max_steps:
            s, a, r, s_, R, mc_R, mean_R, h, p, inits = self.get_batch()

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            returns.append(R)
            mc_returns.append(mc_R)
            mean_Rs.append(mean_R)
            horizons.append(h)
            probs.append(p)
            is_init.append(inits)
            current_steps += 1

        val_losses = []
        for e in range(self.num_value_steps):
            s, R, mean_R, r, h = states[e], mc_returns[e], mean_Rs[e], rewards[e], horizons[e]
            val_loss, _ = self.update_value_function(s, mean_R, R, r, h)
            val_losses.append(val_loss.detach().cpu().numpy())
        self.global_step += self.num_value_steps

        # compute expected returns using value function
        # also add these returns to the PF if possible.
        pf_candidates = []
        for e in range(len(mc_returns)):
            E_Rs = self.value_net(th.FloatTensor(np.array(states[e])).to(self.device),
                                            th.FloatTensor(np.array(mc_returns[e])).to(self.device),
                                            th.FloatTensor(np.array(horizons[e])).unsqueeze(1).to(self.device)).detach().cpu().numpy()
            # add expected return to PF
            for i in range(self.batch_size):
                if is_init[e][i] == 1:
                    pf_candidates.append((E_Rs[i], horizons[e][i]))
            expected_Rs.append(E_Rs)
        self._add_to_pf(pf_candidates)

        policy_losses = []
        for e in range(self.num_policy_steps):
            s, a, R, h, E_Rs, p = states[e], actions[e], returns[e], horizons[e], expected_Rs[e], probs[e]
            policy_loss, _ = self.update_policy(s, a, R, h, E_Rs, p)
            policy_losses.append(policy_loss.detach().cpu().numpy())
        self.global_step += self.num_policy_steps

        popf_losses = []
        if self.use_popf:
            for e in range(self.num_popf_steps):
                prev = expected_Rs[e]
                s, a, r, s_ = states[e], actions[e], rewards[e], next_states[e]
                N = (prev - np.array(r)) / self.gamma
                next_R = self.value_net(th.FloatTensor(np.array(next_states[e])).to(self.device),
                                            th.FloatTensor(np.array(mc_returns[e])).to(self.device),
                                            th.FloatTensor(np.array(horizons[e]) - 1).unsqueeze(1).to(self.device)).detach().cpu().numpy()
                popf_loss, _ = self.update_popf(N, s, a, s_, next_R)
                popf_losses.append(popf_loss.detach().cpu().numpy())
            self.global_step += self.num_popf_steps

        return val_losses, popf_losses, policy_losses

    def train(
            self,
            total_timesteps,
            num_er_episodes,
            num_policy_steps = 1000,
            num_value_steps = 500,
            num_popf_steps = 1000,
            num_expl_episodes = 64,
            num_pf_points = 25,
            log_every = 1,
            prune_pf_every = 25,
            pf_prune_threshold = np.array([1,1]),
            num_eval_iter = 1,
            plot_results = True,
            ):
        """
        Args:
            total_timesteps: Total amount of steps used for training
            num_er_episodes: Number of episodes used to fill replay buffer
            num_policy_steps: Minimum number of policy updates per episode
            num_value_steps: Minimum number of value updates per episode
            num_popf_steps: Minimum number of popf net updates per episode
            num_pf_points: Number of PF points to keep
            prune_pf_every: How many iterations to perform before PF pruning
            log_every: How many iterations to perform before logging/saving
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "num_er_episodes": num_er_episodes,
                    "num_expl_episodes": num_expl_episodes,
                    "num_value_steps": num_value_steps,
                    "num_policy_steps": num_policy_steps,
                    "num_popf_steps": num_popf_steps,
                    "num_points_pf": num_pf_points,
                    "prune_pf_every": prune_pf_every,
                    "pf_prune_threshold": pf_prune_threshold,
                    "num_eval_iter": num_eval_iter,
                }
            )
        self.global_step = 0
        self.total_episodes = num_er_episodes
        self.num_value_steps = num_value_steps
        self.num_popf_steps = num_popf_steps
        self.num_policy_steps = num_policy_steps
        self.num_pf_points = num_pf_points
        n_checkpoints = 0
        evals = 0
        iteration = 0
        self.experience_replay = []
        total_episodes = num_er_episodes

        for _ in range(num_er_episodes):
            transitions = []
            obs,_ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                n_obs, reward, terminated, truncated, _ = self.env.step(action)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), np.float32(reward).copy(), np.float32(reward).copy(), 1, n_obs, 1/self.action_dim, terminated))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
            self._add_episode(transitions, max_size=self.max_buffer_size, step=self.global_step, fill_buffer=True)

        
        while self.global_step < total_timesteps:
            iteration += 1
            val_losses, popf_losses, policy_losses = self.update()

            np.set_printoptions(precision=3)
            print(f"Pareto Front: \n {np.array(self.pf_points, dtype=object)}")

            desired_return, desired_horizon = self._choose_commands()
            expl_returns = []
            for _ in range(num_expl_episodes):
                transitions = self._run_episode(desired_return, desired_horizon)
                self.global_step += len(transitions)
                self._add_episode(transitions, self.max_buffer_size, self.global_step)
                expl_returns.append(transitions[0].return_)
            # update replay buffer with correct distances
            self._update_er_distances()

            # prune non-achievable returns from PF
            eval_pf = []
            if iteration >= (evals + 1) * prune_pf_every:
                evals += 1
                print("Pruning PF")
                eval_pf = self.evaluate_pf(num_eval_iter)
                eval_pf = self.prune_pf(eval_pf, pf_prune_threshold)
            
            if not self.use_popf:
                popf_losses = [0]

            print(
                f"step {self.global_step} \t return {np.mean(expl_returns, axis=0)}, ({np.std(expl_returns, axis=0)}), new return {desired_return} \t value loss {np.mean(val_losses):.3E} \t policy loss {np.mean(policy_losses):.3E} \t popf loss {np.mean(popf_losses):.3E}"
            )

            total_episodes += num_expl_episodes
            
            if iteration >= (n_checkpoints + 1) * log_every:
                n_checkpoints += 1
                script_dir = os.path.dirname(__file__)
                plot_dir = os.path.join(script_dir, 'Results/plots/')
                weights_dir = os.path.join(script_dir, 'Results/weights/')
                data_dir = os.path.join(script_dir, 'Results/data/')
                if not os.path.isdir(plot_dir):
                        os.makedirs(plot_dir)
                if not os.path.isdir(data_dir):
                        os.makedirs(data_dir)
                self.save(n_checkpoints, weights_dir)

                if len(self.pf_points) > 0:
                    er_returns = [r[2][0].return_ for r in self.experience_replay]
                    x1, y1 = zip(*er_returns)
                    pf = [p[0] for p in self.pf_points]
                    if len(eval_pf) == 0:
                        # avoid repeating expensive env steps
                        eval_pf = self.evaluate_pf(num_eval_iter)
                    eval_returns = [p[0] for p in eval_pf]

                    # write PF points to csv file
                    csv_file = data_dir + 'points_' + str(n_checkpoints) + '.csv'
                    header = [
                        "x1",
                        "y1",
                        #"z1",
                        "h1",
                        "x2",
                        "y2",
                        #"z2",
                        "h2"
                    ]
                    with open(csv_file, 'w', newline='') as p:
                        writer = csv.writer(p)
                        writer.writerow(header)
                        for pf_points, eval_points in zip(self.pf_points, eval_pf):
                            ret1, h1 = pf_points
                            ret2, h2 = eval_points
                            row = list(ret1) + [h1] + list(ret2) + [h2]
                            writer.writerow(row)

                    if plot_results:
                        x2, y2 = zip(*pf)
                        x3, y3 = zip(*eval_returns)
                        plt.figure(figsize=(8, 6))
                        plt.scatter(x1, y1, color='blue', label='buffer returns')
                        plt.scatter(x2, y2, color='red', label='current pf')
                        plt.scatter(x3, y3, color='green', label='evaluated pf')

                        # Add labels and legend
                        plt.xlabel('Obj 1')
                        plt.ylabel('Obj 2')
                        plt.legend()
                        plt.title('Comparison of buffer and PF: Checkpoint' + str(n_checkpoints))

                        # Save the figure
                        f = 'training_plot_'+ str(n_checkpoints) + '.png'
                        plt.savefig(plot_dir + f)
                        plt.close()                