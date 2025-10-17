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
    horizon: int
    next_observation: np.ndarray
    terminal: bool
    init: bool

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


class PolicyNet(nn.Module):
    """
    Policy network pi(a|s, R') (actor):
        Only supports discrete actions
    """
    def __init__(self, state_dim: int, reward_dim: int, action_dim: int, scaling_factor, hidden_dim: int = 64):
        super().__init__()
        self.scaling_factor = scaling_factor
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
        s = self.state_emb(state)
        c = th.cat((desired_return, horizon), dim=1) * self.scaling_factor
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

class PVCN(MOAgent, MOPolicy):
    def __init__(
            self,
            env: gym.Env,
            scaling_factor,
            policy_lr: float = 1e-3,
            popf_lr: float = 1e-3,
            gamma: float = 1,
            alpha = .1,
            batch_size: int = 32,
            max_buffer_size: int = 512,
            use_popf: bool = True,
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
            batch_size: Maximum batch size for learning.
            max_buffer_size: Maximum size of ER buffer (= amount of trajectories)
            use_popf: Whether to use POPF network
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        self.experience_replay: List[Transition] = []
        self.state_to_eps = defaultdict(set) # maps a (state, action, horizon) tuple to episode ids which contain that pair
        self.ep_to_transitions = defaultdict(list) # maps an episode id to its corresponding transitions

        self.policy_lr = policy_lr
        self.popf_lr = popf_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.use_popf = use_popf
        self.alpha = alpha
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)

        self.pf_points = [] # tuples (return, horizon)
        self.relaxed_pf_points = [] # backup PF

        # initialize networks
        th.autograd.set_detect_anomaly(True, )
        self.policy = PolicyNet(self.observation_shape[0], self.reward_dim, self.action_dim, self.scaling_factor).to(self.device)

        self.policy_opt = th.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        if use_popf:
            self.popf = PopNet(self.reward_dim + 1 + self.observation_shape[0]*2, self.reward_dim).to(self.device)
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
            "policy_lr": self.policy_lr,
            "popf_lr": self.popf_lr,
            "buffer_size": self.max_buffer_size,
            "use_popf": self.use_popf,
            "seed": self.seed,
        }

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return and add to dictionaries
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].horizon += transitions[i+1].horizon
            transitions[i].return_ += self.gamma * transitions[i + 1].return_

            state = transitions[i+1].observation.tobytes()
            action = transitions[i+1].action
            # add episode id (= step) and index of this transition to the dict
            ep_data = EpisodeStateData(step, i+1)
            key = (state, action, transitions[i+1].horizon)
            self.state_to_eps[key].add(ep_data)
        if len(transitions) > 1:
            # update horizon from initial state!
            transitions[0].horizon = transitions[1].horizon + 1
        # also add first transition of the list
        state = transitions[0].observation.tobytes()
        # NOTE: currently only discrete actions are supported...
        action = transitions[0].action
        self.state_to_eps[(state, action, transitions[0].horizon)].add(EpisodeStateData(step, 0))
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
                a = t.action
                key = (s, a, t.horizon)
                if not key in keys_to_skip:
                    self.state_to_eps[key].remove(old_data)
                    if len(self.state_to_eps[key]) == 0:
                        keys_to_skip.add(key)
                        # remove key from dict if empty list
                        del self.state_to_eps[key]
            del self.ep_to_transitions[old_data.id]
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _update_er_distances(self, threshold=.2):
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
        l2[sma] *= 2
        # update all distances in heap
        for i in range(len(l2)):
            self.experience_replay[i] = (l2[i], self.experience_replay[i][1], self.experience_replay[i][2])
        heapq.heapify(self.experience_replay)

    # TODO: consider adding a secondary PF. This could potentially help in exploration
    def _add_to_pf(self, returns):
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
          sorted_inds = np.argsort(distances)
          # only keep best num_pf_points
          sorted_inds = sorted_inds[-self.num_pf_points:]
          self.pf_points = self.pf_points[sorted_inds].tolist()
        else:
          self.pf_points = self.pf_points.tolist()

    """
        Pick a desired return from the PF and increase it in one objective based on the stdev of the PF returns.
        If only a single return is present, use a user-defined stdev
    """
    """def _choose_commands(self):
        # pick random objective
        r_i = self.np_random.integers(0, self.reward_dim)
        stdev = np.std(np.array([p[0] for p in self.pf_points]), axis=0)[r_i]
        # pick random return
        ret_idx = self.np_random.integers(0, len(self.pf_points))
        desired_return = np.array(self.pf_points[ret_idx], dtype=object)
        # make a deep copy as we are working with np array of objects!
        c = copy.deepcopy(desired_return)
        c[0][r_i] += self.np_random.uniform(0, stdev)
        desired_horizon = c[1]
        # decrease horizon in attempt to improve
        # do not limit to be positive so that model learns this is impossible
        desired_horizon -= 2
        return np.float32(c[0]), np.float32(desired_horizon)"""
    
    def _choose_commands(self):
        ret_idx = self.np_random.integers(0, len(self.pf_points))
        desired_return = np.array(self.pf_points[ret_idx], dtype=object)
        return np.float32(desired_return[0]), np.float32(desired_return[1])

    def update_popf(self, N, states, actions, next_states, next_Rs):
        states = th.FloatTensor(states)
        actions = th.FloatTensor(actions).unsqueeze(1)
        next_states = th.FloatTensor(next_states)
        N = th.FloatTensor(N)
        x = th.cat((N, states, actions, next_states), dim=1).float().to(self.device)
        targets = th.FloatTensor(next_Rs).to(self.device)
        self.popf.train()
        self.popf_opt.zero_grad()

        preds = self.popf(x)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        self.popf_opt.step()
        self.popf_sched.step()
        self.popf.eval()
        return loss, preds

    def compute_policy_loss(self, preds, actions):
        """
        An entropy term is added to ensure exploration
        """
        entropy = -th.sum(preds * th.log(preds + 1e-8), dim=1).mean()
        loss = -th.mean(th.log(preds[th.arange(self.batch_size), actions] + 1e-8)) - self.alpha*entropy
        return loss
    def update_policy(self, states, actions, horizons, expected_returns):
        states = th.FloatTensor(states).to(self.device)
        horizons = th.FloatTensor(horizons).unsqueeze(1).to(self.device)
        expected_returns = th.FloatTensor(expected_returns).to(self.device)

        self.policy_opt.zero_grad()
        preds = self.policy(states, expected_returns, horizons)
        loss = self.compute_policy_loss(preds, actions)
        loss.backward()
        self.policy_opt.step()
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
            action = self.np_random.choice(np.arange(len(probs)), p=probs)
        return action

    def _run_episode(self, desired_return, desired_horizon, eval_mode=False):
        transitions = []
        obs, _ = self.env.reset()
        done = False
        init = True
        N = desired_return
        while not done:
            action = self._act(obs, desired_return, desired_horizon, eval_mode)
            n_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            transitions.append(Transition(obs, action, np.float32(reward).copy(), np.float32(reward).copy(), 1, n_obs, terminated, init))
            init = False

            desired_horizon = np.float32(max(desired_horizon - 1, 1.0)) # avoid neg. horizon
            if self.use_popf:
                N = (N-reward)/self.gamma
                x = np.concatenate((N, obs, [action], n_obs))
                desired_return = self.popf(
                    th.tensor(x).float().to(self.device)
                ).detach().cpu().numpy()
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
        th.save(self.policy, f"{savedir}/policy_{checkpoint}.pt")
        if self.use_popf:
            th.save(self.popf, f"{savedir}/popf_{checkpoint}.pt")

    def get_batch(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        mean_Rs = []
        next_Rs = []
        horizons = []
        is_init = []

        for _ in range(self.batch_size):
            key = random.choice(list(self.state_to_eps.keys()))
            eps_data = np.array(list(self.state_to_eps[key]), dtype=object)
            eps_data = self.np_random.choice(eps_data, self.num_value_samples)
            mean_Rs.append(np.fromiter((self.ep_to_transitions[e.id][e.idx].return_ for e in eps_data), dtype=np.ndarray).mean())
            # compute next value vector
            # find a transition that does not lead to terminal state
            # and compute the next value vector for POPF
            for i in range(self.num_value_samples):
                t: Transition = self.ep_to_transitions[eps_data[i].id][eps_data[i].idx]
                if t.horizon > 1:
                    next_t: Transition = self.ep_to_transitions[eps_data[i].id][eps_data[i].idx+1]
                    next_s = next_t.observation.tobytes()
                    next_key = (next_s, next_t.action, next_t.horizon)
                    eps_data2 = np.array(list(self.state_to_eps[next_key]), dtype=object)
                    eps_data2 = self.np_random.choice(eps_data2, self.num_value_samples)
                    next_Rs.append(np.fromiter((self.ep_to_transitions[e.id][e.idx].return_ for e in eps_data2), dtype=np.ndarray).mean())
                    rewards.append(t.reward)
                    next_states.append(t.next_observation)
                    break
                # If all transitions are terminal => next value vector is 0
                if i == self.num_value_samples-1:
                    rewards.append(t.reward)
                    next_states.append(t.next_observation)
                    next_Rs.append(np.zeros(self.reward_dim))
            # s,a,h are identical
            states.append(t.observation)
            actions.append(np.array(t.action))
            horizons.append(np.array(t.horizon))
            is_init.append(np.fromiter((self.ep_to_transitions[e.id][e.idx].init for e in eps_data), dtype= bool).any())

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(mean_Rs), np.array(next_Rs), np.array(horizons), is_init

    def update(self):
        states = [] # shape = (max_steps, batch_size, obs_dim)
        actions = []
        rewards = []
        next_states = []
        mean_Rs = []
        next_Rs = []
        horizons = []
        is_init = []

        max_steps = max(self.num_policy_steps, self.num_popf_steps)
        for _ in range(max_steps):
            s, a, r, s_, mean_R, next_R, h, inits = self.get_batch()
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            mean_Rs.append(mean_R)
            next_Rs.append(next_R)
            horizons.append(h)
            is_init.append(inits)

        # add expected returns to the PF if Pareto dominant
        pf_candidates = []
        for b in range(max_steps):
            for i in range(self.batch_size):
                if is_init[b][i]:
                    pf_candidates.append((mean_Rs[b][i], horizons[b][i]))
        self._add_to_pf(pf_candidates)

        policy_losses = []
        for b in range(self.num_policy_steps):
            s, a, h, E_Rs = states[b], actions[b], horizons[b], mean_Rs[b]
            policy_loss, _ = self.update_policy(s, a, h, E_Rs)
            policy_losses.append(policy_loss.detach().cpu().numpy())
        self.global_step += self.num_policy_steps

        popf_losses = []
        
        if self.use_popf:
            for i in range(self.num_popf_steps):
                prev = mean_Rs[i]
                s, a, r, s_ = states[i], actions[i], rewards[i], next_states[i]
                N = (prev - r) / self.gamma
                next_R = next_Rs[i]
                popf_loss, _ = self.update_popf(N, s, a, s_, next_R)
                popf_losses.append(popf_loss.detach().cpu().numpy())
            self.global_step += self.num_popf_steps

        return popf_losses, policy_losses

    def train(
            self,
            total_timesteps,
            num_er_episodes,
            num_value_samples = 10,
            num_policy_steps = 1000,
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
            num_policy_steps: Number of policy updates per episode
            num_value_samples: Number of samples used for computing mean return
            num_popf_steps: Minimum number of popf net updates per episode
            num_pf_points: Number of PF points to keep
            prune_pf_every: How many iterations to perform before PF pruning
            log_every: How many iterations to perform before logging/saving
            num_eval_iter: Iterations to perform for evaluating PF points
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "num_er_episodes": num_er_episodes,
                    "num_expl_episodes": num_expl_episodes,
                    "num_policy_steps": num_policy_steps,
                    "num_popf_steps": num_popf_steps,
                    "num_points_pf": num_pf_points,
                    "prune_pf_every": prune_pf_every,
                    "pf_prune_threshold": pf_prune_threshold,
                    "num_eval_iter": num_eval_iter,
                }
            )
        self.global_step = 0
        self.num_value_samples = num_value_samples
        self.total_episodes = num_er_episodes
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
            init = True
            while not done:
                action = self.env.action_space.sample()
                n_obs, reward, terminated, truncated, _ = self.env.step(action)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), np.float32(reward).copy(), 1, n_obs, terminated, init))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
                init = False
            self._add_episode(transitions, max_size=self.max_buffer_size, step=self.global_step)

        
        while self.global_step < total_timesteps:
            iteration += 1
            popf_losses, policy_losses = self.update()

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
                f"step {self.global_step} \t return {np.mean(expl_returns, axis=0)}, ({np.std(expl_returns, axis=0)}), new return {desired_return} \t policy loss {np.mean(policy_losses):.3E} \t popf loss {np.mean(popf_losses):.3E}"
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