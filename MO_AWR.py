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


@dataclass
class Transition:
    """Transition dataclass."""
    observation: np.ndarray
    action: Union[float, int]
    reward: np.ndarray
    return_: np.ndarray
    value_vec: np.ndarray
    horizon: int
    next_observation: np.ndarray
    action_prob: float
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
            nn.LogSoftmax(dim=1)
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
    It is part of the actor as it the suggested desired value vector is used by it.
    Code from: https://github.com/rradules/POP-following
    """

    def __init__(self, d_in, d_out, dropout=0.5):
        super(PopNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, d_out),
        )
    def forward(self, x):
        return self.layers(x)

class MO_AWR(MOAgent, MOPolicy):
    def __init__(
            self,
            env: gym.Env,
            policy_lr: float = 1e-4,
            value_lr: float = 1e-3,
            popf_lr: float = 1e-3,
            gamma: float = 1,
            td_lambda: float = 0.95,
            beta: float = 0.1,
            cd_threshold: float = 0.2,
            batch_size: int = 32,
            max_buffer_size: int = 1024,
            max_return: Optional[np.ndarray] = None,
            use_popf: bool = True,
            use_critic_msbe=True,
            device: Union[th.device, str] = "auto",
            log: bool = False,
            wandb_entity: Optional[str] = None,
            project_name: str = "MORL-Baselines",
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
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        self.experience_replay: List[Transition] = []
        self.state_to_eps = defaultdict(set) # maps a state-action pair to episode ids which contain that pair
        self.ep_to_transitions = defaultdict(list) # maps an episode id to its corresponding transitions

        self.max_cd_value = self.reward_dim

        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.popf_lr = popf_lr
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.beta = beta
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.cd_threshold = cd_threshold
        self.use_popf = use_popf
        self.max_return = max_return
        self.use_critic_msbe = use_critic_msbe

        self.pf_points = [] # tuples (return, horizon)

        # initialize networks
        th.autograd.set_detect_anomaly(True, )
        self.value_net = ValueNet(self.observation_dim, self.reward_dim)
        self.policy = PolicyNet(self.observation_dim, self.reward_dim, self.action_dim)

        #th.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        #th.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)


        self.val_opt = th.optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_opt = th.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        if use_popf:
            self.popf = PopNet(self.reward_dim + 1 + self.observation_dim*2, self.reward_dim)
            self.popf_opt = th.optim.Adam(self.popf.parameters(), lr=self.popf_lr)

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
            "beta": self.beta,
            "weight_clip": self.weight_clip,
            "buffer_size": self.max_buffer_size,
            "use_popf": self.use_popf,
            "cd_threshold": self.cd_threshold,
            "seed": self.seed,
        }

    # code from https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/multi_policy/pcn/pcn.py
    def _add_episode(self, transitions: List[Transition], max_size: int, step: int, fill_buffer=False) -> None:
        # compute return and add to dictionaries
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].horizon += transitions[i+1].horizon
            if (not fill_buffer) and self.td_lambda < 1:
                transitions[i].return_ = transitions[i].reward + self.gamma * ((1-self.td_lambda) *
                                                                               self.value_net(th.FloatTensor(transitions[i].observation).unsqueeze(0).to(self.device),
                                                                                              th.FloatTensor(transitions[i].value_vec).unsqueeze(0).to(self.device),
                                                                                              th.FloatTensor([transitions[i].horizon]).unsqueeze(0).to(self.device)).detach().numpy()[0] +
                                                                               self.td_lambda*transitions[i + 1].return_)
            else:
                transitions[i].return_ += self.gamma * transitions[i + 1].return_

            state = transitions[i+1].observation.tobytes()
            # add episode id (= step) and index of this transition to the dict
            ep_data = EpisodeStateData(step, i+1)
            key = (state, transitions[i+1].action)
            self.state_to_eps[key].add(ep_data)
        if len(transitions) > 1:
            # update horizon from initial state!
            transitions[0].horizon = transitions[1].horizon + 1
        # also add first transition of the list
        self.state_to_eps[(transitions[0].observation.tobytes(), transitions[0].action)].add(EpisodeStateData(step, 0))
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
                s = t.observation.tobytes()
                key = (s, t.action)
                if not key in keys_to_skip:
                    self.state_to_eps[key].remove(old_data)
                    if len(self.state_to_eps[key]) == 0:
                        keys_to_skip.add(key)
                        # remove key from dict if empty list
                        del self.state_to_eps[key]
            del self.ep_to_transitions[old_data.id]
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _update_er_distances(self, threshold, keep_duplicates=False):
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
        nd_inds = get_non_dominated_inds(np.float32([ret[0] for ret in self.pf_points]))
        # Only keep ND points
        self.pf_points = np.array(self.pf_points, dtype=object)[nd_inds]
        distances = crowding_distance(np.array([nd[0] for nd in self.pf_points], dtype=object))
        sorted_inds = np.argsort(distances) # NOTE: stable arg not supported in this np version...
        # only keep best num_pf_points
        sorted_inds = sorted_inds[-self.num_pf_points:]
        self.pf_points = self.pf_points[sorted_inds].tolist()

    def _choose_commands(self, num_returns):
        """
        Pick a desired return from the PF and increase it in one objective:
            - num_returns determines the amount of returns considered for computation of the stdev
            - If num_returns < size of PF, compute stdev over whole PF
        """
        # pick random objective
        r_i = self.np_random.integers(0, self.reward_dim)
        # pick random returns
        if num_returns < len(self.pf_points):
            returns_i = self.np_random.choice(np.arange(len(self.pf_points)), num_returns)
            pf_points = self.pf_points[returns_i]
            stdev = np.std(np.array([p[0] for p in pf_points]), axis=0)[r_i] # compute stdev for objective r_i
            desired_return = self.np_random.choice(pf_points)
        else:
            ret_idx = self.np_random.integers(0, len(self.pf_points))
            stdev = np.std(np.array([p[0] for p in self.pf_points]), axis=0)[r_i]
            desired_return = np.array(self.pf_points[ret_idx], dtype=object)
        # make a deep copy as we are working with np array of objects!
        c = copy.deepcopy(desired_return)
        c[0][r_i] += self.np_random.uniform(0, stdev)
        desired_horizon = c[1]
        # decrease horizon in attempt to improve
        desired_horizon -= 2

        return np.float32(c[0]), np.float32(desired_horizon)

    def compute_value_loss(self, preds, targets, mean_rs):
        """
        Idea:
            - Compute SE between targets and mean_rs
            - Compute squared error between each pred and target
            - Weight each error term using 1/(MSE(targets, mean_rs)+1). This will push the value estimates towards the desired returns while keeping sample-efficiency
            - The loss behaves like MSBE for returns ~= desired return
        """
        weights = 1/(th.linalg.vector_norm(targets - mean_rs, dim=1) + 1)**2
        errors = th.linalg.vector_norm(preds - targets, dim=1)**2
        return th.mean(errors*weights)

    def update_value_function(self, states, mean_r, returns, horizons):
        mean_r = th.FloatTensor(np.full((self.batch_size, self.reward_dim), mean_r)).to(self.device)
        states = th.FloatTensor(np.array(states)).to(self.device)
        returns = th.FloatTensor(np.array(returns))
        horizons = th.FloatTensor(np.array(horizons)).unsqueeze(1).to(self.device)

        self.value_net.train()
        self.val_opt.zero_grad()
        preds = self.value_net(states, mean_r, horizons)

        if self.use_critic_msbe:
            loss = F.mse_loss(preds, returns)
        else:
            loss = self.compute_value_loss(preds, returns, mean_r)
        loss.backward()
        self.val_opt.step()
        self.value_net.eval()
        return loss, preds

    def update_popf(self, N, states, actions, next_states, next_R):
        states = th.FloatTensor(np.array(states))
        actions = th.unsqueeze(th.tensor(actions), 1)
        next_states = th.FloatTensor(np.array(next_states))
        N = th.tensor(np.float32(N))
        x = th.cat((N, states, actions, next_states), dim=1).float().to(self.device)
        targets = th.tensor(np.full((self.batch_size, self.reward_dim), next_R)).float()
        self.popf_opt.zero_grad()

        preds = self.popf(x)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        self.popf_opt.step()
        return loss, preds

    def _dominates(self, a, b):
        return np.all(a >= b) and np.any(a > b)
    def compute_advantages(self, states, returns, mean_Rs, horizons):
        """
        Computed advantage weights for policy loss:
            - If the return dominates the desired return, use L2-norm
            - Else, use the neg. L2-norm
        """
        weights = []
        V_s = self.value_net(states, mean_Rs, horizons).detach().cpu().numpy()
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
    def compute_policy_loss(self, preds, actions, states, returns, mean_rs, horizons, probs):
        """
        Loss is computed using advantage estimates. Each loss is also weighted using importance sampling ratios
        """
        advantages = self.compute_advantages(states, returns, mean_rs, horizons)
        # We already us logsoftmax activation so log does not need to be computed here.
        loss = -th.mean(preds[:, actions]*th.exp(advantages/self.beta))
        return loss
    def update_policy(self, states, actions, returns, mean_r, horizons, probs):
        mean_r = th.FloatTensor(np.full((self.batch_size, self.reward_dim), mean_r)).to(self.device)
        states = th.FloatTensor(np.array(states)).to(self.device)
        horizons = th.FloatTensor(np.array(horizons)).unsqueeze(1).to(self.device)

        self.policy_opt.zero_grad()
        preds = self.policy(states, mean_r, horizons)
        loss = self.compute_policy_loss(preds, actions, states, returns, mean_r, horizons, probs)
        loss.backward()
        self.policy_opt.step()
        return loss, preds

    def _act(self, state, desired_return, desired_horizon, eval_mode=False):
        pred = self.policy(
            th.unsqueeze(th.FloatTensor(state), 0).to(self.device),
            th.unsqueeze(th.FloatTensor(desired_return), 0).to(self.device),
            th.unsqueeze(th.FloatTensor([desired_horizon]), 0).to(self.device),
        )
        log_probs = pred.detach().cpu().numpy()[0]

        if eval_mode:
            action = np.argmax(log_probs)
        else:
            action = self.np_random.choice(np.arange(len(log_probs)), p=np.exp(log_probs)) # exponentiate to get proper prob.
        return action, np.exp(log_probs[action])

    def _run_episode(self, desired_return, desired_horizon, eval_mode=False):
        transitions = []
        obs, _ = self.env.reset()
        done = False
        N = desired_return
        while not done:
            action, prob = self._act(obs, desired_return, desired_horizon, eval_mode)
            n_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            transitions.append(Transition(obs, action, np.float32(reward).copy(), np.float32(reward).copy(), desired_return, 1, n_obs, prob, terminated))

            if self.use_popf:
                N = (N-reward)/self.gamma
                x = np.concatenate((N, obs, [action], n_obs))
                desired_return = self.popf(
                    th.tensor(x).float().to(self.device)
                )
            else:
                desired_return = np.clip(desired_return - reward, -np.inf, self.max_return, dtype=np.float32)

            desired_horizon = np.float32(max(desired_horizon - 1, 1.0)) # avoid neg. horizon
            obs = n_obs
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
        If the difference between a point and its evaluation < threshold, keep it in the PF
        """
        new_pf = []
        for i in range(len(eval_pf)):
            diff = np.absolute(eval_pf[i][0] - self.pf_points[i][0])
            if np.all(diff < threshold):
                new_pf.append(self.pf_points[i])
            else:
                new_pf.append(eval_pf[i])
        self.pf_points = new_pf 

    def save(self, savedir: str = "weights"):
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        th.save(self.value_net, f"{savedir}/value.pt")
        th.save(self.value_net, f"{savedir}/policy.pt")
        if self.use_popf:
            th.save(self.value_net, f"{savedir}/popf.pt")

    def get_batch(self, state, action):
        """
        Get batch of (s, a, r, s', R, R^, H)
        s and a are the same per batch
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        returns = []
        mean_R = np.zeros(self.reward_dim)
        horizons = []
        probs = []
        # sample episodes that visit state-action pair
        key = (state.tobytes(), action)
        eps_data = np.array(list(self.state_to_eps[key]), dtype=object)
        eps_data = self.np_random.choice(eps_data, self.batch_size)

        for e in eps_data:
            t: Transition = self.ep_to_transitions[e.id][e.idx]
            states.append(t.observation)
            actions.append(t.action)
            rewards.append(np.float32(t.reward))
            next_states.append(t.next_observation)
            returns.append(np.float32(t.return_))
            mean_R += np.float32(t.return_)
            horizons.append(np.float32(t.horizon))
            probs.append(np.float32(t.action_prob))
        return states, actions, rewards, next_states, returns, mean_R/self.batch_size, horizons, probs

    def update(self):
        states = [] # shape = (batch_size, obs_dim)
        actions = []
        rewards = []
        next_states = []
        returns = []
        mean_Rs = []
        horizons = []
        probs = []

        pf_candidates = []

        max_steps = max(self.num_policy_steps, self.num_value_steps, self.num_popf_steps)
        current_steps = 0
        init_state, _ = self.env.reset()
        while current_steps < max_steps:
            # pick random state-action pair from er
            # cannot use np_random here because it does not support bytes
            k = random.choice(list(self.state_to_eps.keys()))
            state = np.frombuffer(k[0], dtype=np.int32).reshape(self.observation_shape)
            s, a, r, s_, R, mean_R, h, p = self.get_batch(state, k[1])
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            returns.append(R)
            mean_Rs.append(mean_R)
            horizons.append(h)
            probs.append(p)
            current_steps += 1
            if np.array_equal(init_state, state):
                pf_candidates.append((mean_R, np.mean(h)))
        if len(pf_candidates) > 0:
            self._add_to_pf(pf_candidates)

        print('training value net')
        val_losses = []
        for i in range(self.num_value_steps):
            s, R, mean_R, h = states[i], returns[i], mean_Rs[i], horizons[i]
            val_loss, _ = self.update_value_function(s, mean_R, R, h)
            val_losses.append(val_loss.detach().cpu().numpy())
        self.global_step += self.num_value_steps

        popf_losses = []
        if self.use_popf:
            popf_steps = 0
            print('training POPF net')
            for e in range(len(states)):
                prev = mean_Rs[e][0]
                for i in range(len(states[e])):
                    s, a, r, s_, next_R = states[e][i], actions[e][i], rewards[e][i], next_states[e][i], mean_Rs[e][i]
                    N = (prev - r) / self.gamma
                    popf_loss, _ = self.update_popf(N, s, a, s_, next_R)
                    popf_losses.append(popf_loss.detach().cpu().numpy())
                    prev = next_R
                    popf_steps += len(states[e])
                if popf_steps > self.num_popf_steps:
                    break;
            self.global_step += popf_steps

        policy_losses = []
        print('training policy')
        for i in range(self.num_policy_steps):
            prev = mean_Rs[0]
            s, a, R, mean_R, h, p = states[i], actions[i], returns[i], mean_Rs[i], horizons[i], probs[i]
            policy_loss, _ = self.update_policy(s, a, R, mean_R, h, p)
            policy_losses.append(policy_loss.detach().cpu().numpy())
        self.global_step += self.num_policy_steps

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
            prune_pf_every = 5,
            pf_prune_threshold = np.array([1,1]),
            num_eval_iter = 1
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
                    "max_buffer_size": self.max_buffer_size,
                    "num_points_pf": num_pf_points,
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
                transitions.append(Transition(obs, action, np.float32(reward).copy(), np.float32(reward).copy(), np.zeros(self.reward_dim), 1, n_obs, 1/self.action_dim, terminated))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
            self._add_episode(transitions, max_size=self.max_buffer_size, step=self.global_step, fill_buffer=True)

        
        while self.global_step < total_timesteps:
            iteration += 1
            val_losses, popf_losses, policy_losses = self.update()

            np.set_printoptions(precision=3)
            print(f"Pareto Front: {np.array(self.pf_points, dtype=object)}")

            desired_return, desired_horizon = self._choose_commands(num_pf_points) # TODO: decide on what % of the PF to use for computing the stdev
            expl_returns = []
            for _ in range(num_expl_episodes):
                transitions = self._run_episode(desired_return, desired_horizon)
                self.global_step += len(transitions)
                self._add_episode(transitions, self.max_buffer_size, self.global_step)
                expl_returns.append(transitions[0].return_)
            # update replay buffer with correct distances
            self._update_er_distances(self.cd_threshold)

            # prune non-achievable returns from PF
            if iteration >= (evals + 1) * prune_pf_every:
                evals += 1
                print("Pruning PF")
                eval_returns = self.evaluate_pf(num_eval_iter)
                self.prune_pf(eval_returns, pf_prune_threshold)
            
            if not self.use_popf:
                popf_losses = [0]

            print(
                f"step {self.global_step} \t return {np.mean(expl_returns, axis=0)}, ({np.std(expl_returns, axis=0)}), new return {desired_return} \t value loss {np.mean(val_losses):.3E} \t policy loss {np.mean(policy_losses):.3E} \t popf loss {np.mean(popf_losses):.3E}"
            )

            total_episodes += num_expl_episodes
            if self.log:
                wandb.log(
                    {
                        "train/episode": total_episodes,
                        "global_step": self.global_step,
                    },
                )

                for i in range(self.reward_dim):
                    wandb.log(
                        {
                            f"train/desired_return_{i}": desired_return[i],
                            f"train/mean_return_{i}": np.mean(np.array(expl_returns)[:, i]),
                            f"train/mean_return_distance_{i}": np.linalg.norm(
                                np.mean(np.array(expl_returns)[:, i]) - desired_return[i]
                            ),
                            "global_step": self.global_step,
                        },
                    )
            
            if iteration >= (n_checkpoints + 1) * log_every:
                self.save()
                n_checkpoints += 1
                er_returns = [r[2][0].return_ for r in self.experience_replay]
                eval_returns = self.evaluate_pf(num_eval_iter)
                x1, y1 = zip(*er_returns)
                x2, y2 = zip(*[p[0] for p in self.pf_points])
                x3, y3 = zip(*[p[0] for p in eval_returns])
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
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'Results/')
                f = 'training_plot_'+ str(n_checkpoints) + '.png'

                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                plt.savefig(results_dir + f)
                plt.close()

                