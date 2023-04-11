from lifelong_rl.core import logger
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from collections import OrderedDict
import time

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.torch.distributions import TanhNormal
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch

# from torch_batch_svd import svd

ACTION_MIN = -1.0
ACTION_MAX = 1.0


class SACTrainer(TorchTrainer):
    """
    Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
    Continuous maximum Q-learning algorithm with parameterized actor.
    """
    def __init__(
            self,
            env,  # Associated environment for learning
            policy,  # Associated policy (should be TanhGaussian)
            qfs,  # Q functions
            target_qfs,  # Slow updater to Q functions
            discount=0.99,  # Discount factor
            reward_scale=1.0,  # Scaling of rewards to modulate entropy bonus
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,  # Target entropy for entropy-constraint variant
            policy_lr=3e-4,  # Learning rate of policy and entropy weight
            qf_lr=3e-4,  # Learning rate of Q functions
            optimizer_class=optim.Adam,  # Class of optimizer for all networks
            soft_target_tau=5e-3,  # Rate of update of target networks
            target_update_period=1,  # How often to update target networks
            max_q_backup=False,
            deterministic_backup=False,
            policy_eval_start=0,
            eta=-1.0,
            num_qs=10,
            replay_buffer=None,
            imagination=None,

            # smoothing
            num_samples=20,
            policy_smooth_eps=0,
            policy_smooth_reg=0.0,
            q_smooth_eps=0,
            q_smooth_reg=0.0,
            q_smooth_tau=0.2,
            norm_input=False,
            obs_std=1,
            q_ood_eps=0,
            q_ood_reg = 0,
            q_ood_uncertainty_reg=0,
            q_ood_uncertainty_reg_min=0,
            q_ood_uncertainty_decay=1e-6,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.num_qs = num_qs

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.eta = eta
        self.replay_buffer = replay_buffer

        #### robust 
        self.num_samples = num_samples
        self.policy_smooth_eps = policy_smooth_eps
        self.policy_smooth_reg = policy_smooth_reg
        self.q_smooth_eps = q_smooth_eps
        self.q_smooth_reg = q_smooth_reg
        self.q_smooth_tau = q_smooth_tau
        self.obs_std = 1 if not norm_input else ptu.from_numpy(obs_std)
        self.q_ood_eps = q_ood_eps
        self.q_ood_reg = q_ood_reg
        self.q_ood_uncertainty_reg = q_ood_uncertainty_reg
        self.q_ood_uncertainty_reg_min = q_ood_uncertainty_reg_min
        self.q_ood_uncertainty_decay = q_ood_uncertainty_decay

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic value: dimension of action space
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(-1, obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions,
                                           1).view(obs.shape[0] * num_actions,
                                                   obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp,
            reparameterize=True,
            return_log_prob=True,
        )
        return new_obs_actions.detach(), new_obs_log_pi.view(
            obs.shape[0], num_actions, 1).detach()
        

    def _get_noised_obs(self, obs, actions, eps):
        M, N, A = obs.shape[0], obs.shape[1], actions.shape[1]
        size = self.num_samples
        delta_s = 2 * eps * self.obs_std * (torch.rand(size, N, device=ptu.device) - 0.5) 
        tmp_obs = obs.reshape(-1, 1, N).repeat(1, size, 1).reshape(-1, N)
        delta_s = delta_s.reshape(1, size, N).repeat(M, 1, 1).reshape(-1, N)
        noised_obs = tmp_obs + delta_s
        return M, A, size, noised_obs, delta_s


    def train_from_torch(self, batch, indices):
        obs= batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        
        if self.eta > 0:
            actions.requires_grad_(True)
        
        """
        Policy and Alpha Loss
        """
        # (B,A)= (256,6)             
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        # (B,1)
        q_new_actions = self.qfs.sample(obs, new_obs_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._num_train_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.get_log_probs(obs.detach(), actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        
        if self.policy_smooth_eps > 0 and self.policy_smooth_reg > 0:
            M, A, size, noised_obs, delta_s = self._get_noised_obs(obs, actions, self.policy_smooth_eps)
            # (B*size,A)
            _, noised_policy_mean, noised_policy_log_std, _, *_ = self.policy(noised_obs,reparameterize=True)
            action_dist = torch.distributions.Normal(policy_mean.reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A), policy_log_std.exp().reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A))
            noised_action_dist = torch.distributions.Normal(noised_policy_mean, noised_policy_log_std.exp())
            kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) + kl_divergence(noised_action_dist, action_dist).sum(axis=-1)
            kl_loss = kl_loss.reshape(M, size)
            max_id = torch.argmax(kl_loss, axis=1)
            kl_loss_max = kl_loss[np.arange(M), max_id].mean()
            # noised_states_selected = noised_obs[np.arange(M), max_id]
            policy_loss += self.policy_smooth_reg * kl_loss_max
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Policy Smooth Loss'] = ptu.get_numpy(kl_loss_max) * self.policy_smooth_reg

        """
        QF Loss
        """
        # (num_qs, batch_size, output_size) (M,B,1)
        qs_pred = self.qfs(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs,
            reparameterize=False,
            return_log_prob=True,
        )

        if not self.max_q_backup:
            target_q_values = self.target_qfs.sample(next_obs, new_next_actions)
            if not self.deterministic_backup:
                target_q_values -= alpha * new_log_pi
        else:
            # if self.max_q_backup
            next_actions_temp, _ = self._get_policy_actions(
                next_obs, num_actions=10, network=self.policy)
            target_q_values = self._get_tensor_values(
                next_obs, next_actions_temp,
                network=self.qfs).max(2)[0].min(0)[0]

        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values

        q_target_detach = q_target.detach().unsqueeze(0).repeat(self.num_qs, 1, 1)
        qfs_loss = self.qf_criterion(qs_pred, q_target_detach)
        qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()
        qfs_loss_total = qfs_loss

        if self.q_smooth_eps > 0 and self.q_smooth_reg > 0:
            M, A, size, noised_obs, delta_s = self._get_noised_obs(obs, actions, self.q_smooth_eps)
            # (M,B*size,1)
            noised_qs_pred = self.qfs(noised_obs, actions.reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A))
            diff = noised_qs_pred - qs_pred.repeat(1, 1, size).reshape(self.num_qs, -1, 1)
            zero_tensor = torch.zeros(diff.shape, device=ptu.device)
            pos, neg = torch.maximum(diff, zero_tensor), torch.minimum(diff, zero_tensor)
            noise_Q_loss = (1-self.q_smooth_tau) *  pos.square().mean(axis=0) + self.q_smooth_tau * neg.square().mean(axis=0)
            noise_Q_loss = noise_Q_loss.reshape(M, size)
            noise_Q_loss_max = noise_Q_loss[np.arange(M), torch.argmax(noise_Q_loss, axis=-1)].mean()
            qfs_loss_total += self.q_smooth_reg * noise_Q_loss_max
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q Smooth Loss'] = ptu.get_numpy(noise_Q_loss_max) * self.q_smooth_reg
                
        if self.q_ood_reg > 0: # self.q_ood_eps = 0 for PBRL
            ood_loss = torch.zeros(1, device=ptu.device)[0]
            if self.q_ood_uncertainty_reg > 0:
                M, A, size, noised_obs, delta_s = self._get_noised_obs(obs, actions, self.q_ood_eps)
                ood_actions, _, _, _, *_ = self.policy(noised_obs, reparameterize=False)
                ood_qs_pred = self.qfs(noised_obs, ood_actions)
                ood_target = ood_qs_pred - self.q_ood_uncertainty_reg * ood_qs_pred.std(axis=0)
                ood_loss = self.qf_criterion(ood_target.detach(), ood_qs_pred).mean()
                qfs_loss_total += self.q_ood_reg * ood_loss

            if self.q_ood_uncertainty_reg > 0:
                self.q_ood_uncertainty_reg = max(self.q_ood_uncertainty_reg - self.q_ood_uncertainty_decay, self.q_ood_uncertainty_reg_min)
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q OOD Loss'] = ptu.get_numpy(ood_loss) * self.q_ood_reg
                self.eval_statistics['q_ood_uncertainty_reg'] = self.q_ood_uncertainty_reg
        
        if self.eta > 0:
            qs_pred_grads = None
            sample_size = min(qs_pred.size(0), actions.size(1))
            indices = np.random.choice(qs_pred.size(0), size=sample_size, replace=False)
            indices = torch.from_numpy(indices).long().to(ptu.device)

            obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
            actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
            qs_preds_tile = self.qfs(obs_tile, actions_tile)
            qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)

            qs_pred_grads = torch.index_select(qs_pred_grads, dim=0, index=indices).transpose(0, 1)
            
            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            masks = torch.eye(sample_size, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            qs_pred_grads = (1 - masks) * qs_pred_grads
            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (sample_size - 1)
            
            qfs_loss_total += self.eta * grad_loss

        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()

        self.try_update_target_networks()
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = ptu.get_numpy(log_pi - q_new_actions).mean()
            policy_avg_std = ptu.get_numpy(torch.exp(policy_log_std)).mean()
            self.eval_statistics['QFs Loss'] = np.mean(
                ptu.get_numpy(qfs_loss)) / self.num_qs
            if self.eta > 0:
                self.eval_statistics['Q Grad Loss'] = np.mean(
                    ptu.get_numpy(grad_loss))
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss)

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Predictions',
                    ptu.get_numpy(qs_pred),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Targets',
                    ptu.get_numpy(q_target),
                ))

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            self.eval_statistics['Policy std'] = np.mean(policy_avg_std)

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs, self.target_qfs,
                                self.soft_target_tau)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.qfs,
            log_alpha=self.log_alpha,
            policy_optim=self.policy_optimizer,
            qfs_optim=self.qfs_optimizer,
            alpha_optim=self.alpha_optimizer,
        )
    
    def load_snapshot(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.policy.load_state_dict(datas['trainer/policy'].state_dict())
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        self.target_qfs.load_state_dict(datas['trainer/target_qfs'].state_dict())
        self.log_alpha = datas['trainer/log_alpha']
        self.policy_optimizer.load_state_dict( datas['trainer/policy_optim'].state_dict())
        self.qfs_optimizer.load_state_dict(datas['trainer/qfs_optim'].state_dict())
        self.alpha_optimizer.load_state_dict(datas['trainer/alpha_optim'].state_dict())
        logger.log('Loading model from {} finished'.format(path))
    
    def load_qfs(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        logger.log('Loading Q networks from {} finished'.format(path))




class SACTrainerRank1(TorchTrainer):
    """
    Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
    Continuous maximum Q-learning algorithm with parameterized actor.
    """
    def __init__(
            self,
            env,  # Associated environment for learning
            policy,  # Associated policy (should be TanhGaussian)
            qfs,  # Q functions
            target_qfs,  # Slow updater to Q functions
            discount=0.99,  # Discount factor
            reward_scale=1.0,  # Scaling of rewards to modulate entropy bonus
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,  # Target entropy for entropy-constraint variant
            policy_lr=3e-4,  # Learning rate of policy and entropy weight
            qf_lr=3e-4,  # Learning rate of Q functions
            optimizer_class=optim.Adam,  # Class of optimizer for all networks
            soft_target_tau=5e-3,  # Rate of update of target networks
            target_update_period=1,  # How often to update target networks
            max_q_backup=False,
            deterministic_backup=False,
            policy_eval_start=0,
            eta=-1.0,
            num_qs=10,
            replay_buffer=None,
            imagination=None,

            # smoothing
            num_samples=20,
            policy_smooth_eps=0,
            policy_smooth_reg=0.0,
            q_smooth_eps=0,
            q_smooth_reg=0.0,
            q_smooth_tau=0.2,
            norm_input=False,
            obs_std=1,
            q_ood_eps=0,
            q_ood_reg = 0,
            q_ood_uncertainty_reg=0,
            q_ood_uncertainty_reg_min=0,
            q_ood_uncertainty_decay=1e-6,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.target_qfs.eval()
        for param in target_qfs.parameters():
            param.requires_grad = False

        self.num_qs = num_qs

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.eta = eta
        self.replay_buffer = replay_buffer

        #### robust 
        self.num_samples = num_samples
        self.policy_smooth_eps = policy_smooth_eps
        self.policy_smooth_reg = policy_smooth_reg
        self.q_smooth_eps = q_smooth_eps
        self.q_smooth_reg = q_smooth_reg
        self.q_smooth_tau = q_smooth_tau
        self.obs_std = 1 if not norm_input else ptu.from_numpy(obs_std)
        self.q_ood_eps = q_ood_eps
        self.q_ood_reg = q_ood_reg
        self.q_ood_uncertainty_reg = q_ood_uncertainty_reg
        self.q_ood_uncertainty_reg_min = q_ood_uncertainty_reg_min
        self.q_ood_uncertainty_decay = q_ood_uncertainty_decay

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic value: dimension of action space
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        # Add weight decay
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
            weight_decay=1e-4,
        ) 

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        B,E = obs.shape
        num_repeat = action_shape // B
        preds = network(obs.unsqueeze(1).expand(-1,num_repeat,-1).contiguous().view(-1,E), actions)
        # preds size (ensembles, B*Num_repeat,E) => (ensembles, B, Num_repeat, E)
        preds = preds.view(-1, B, num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs.torch.repeat_interleave(num_actions, dim=0),
            reparameterize=True,
            return_log_prob=True,
        )
        return new_obs_actions.detach(), new_obs_log_pi.view(
            obs.shape[0], num_actions, 1).detach()
        

    def _get_noised_obs(self, obs, actions, eps):
        """ Add noise to observations 
            Each observation is repeated interleave num_samples times
        """
        M, N = obs.shape[0], obs.shape[1] 
        A = actions.shape[1]
        size = self.num_samples
        delta_s = 2 * eps * self.obs_std * (torch.rand(size, N, device=ptu.device) - 0.5) 
        tmp_obs = obs.repeat_interleave(size,0).view(M,size,N)  
        noised_obs = tmp_obs + delta_s
        return M, A, size, noised_obs.view(-1,N)


    def train_from_torch(self, batch, indices):
        obs= batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        
        if self.eta > 0:
            actions.requires_grad_(True)
        
        """
        Policy and Alpha Loss
        """
        # (B,A)= (256,6)             
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        # (B,1)
        q_new_actions = self.qfs.sample(obs, new_obs_actions)  # TODO: Consider to detach Q value

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._num_train_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.get_log_probs(obs.detach(), actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        
        if self.policy_smooth_eps > 0 and self.policy_smooth_reg > 0:
            M, A, size, noised_obs = self._get_noised_obs(obs, actions, self.policy_smooth_eps)
            # (B*size,A)
            _, noised_policy_mean, noised_policy_log_std, _, *_ = self.policy(noised_obs,reparameterize=True)
            action_dist = torch.distributions.Normal(policy_mean.repeat_interleave(size, dim=0),
                                                    policy_log_std.exp().repeat_interleave(size, dim=0))
                                                    
            noised_action_dist = torch.distributions.Normal(noised_policy_mean, noised_policy_log_std.exp())
            kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) + kl_divergence(noised_action_dist, action_dist).sum(axis=-1)
            kl_loss = kl_loss.view(M, size)
            max_values, _ = torch.max(kl_loss, dim=1)
            kl_loss_max = torch.mean(max_values)



            # noised_states_selected = noised_obs[np.arange(M), max_id]
            policy_loss += self.policy_smooth_reg * kl_loss_max
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Policy Smooth Loss'] = ptu.get_numpy(kl_loss_max) * self.policy_smooth_reg

        """
        QF Loss
        """
        # (num_qs, batch_size, output_size) (M,B,1)
        qs_pred = self.qfs(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs,
            reparameterize=False,
            return_log_prob=True,
        )

        if not self.max_q_backup:
            with torch.no_grad():
                target_q_values = self.target_qfs.sample(next_obs, new_next_actions)
                if not self.deterministic_backup:
                    target_q_values -= alpha * new_log_pi
        else:
            # if self.max_q_backup
            next_actions_temp, _ = self._get_policy_actions(
                next_obs, num_actions=10, network=self.policy)
            target_q_values = self._get_tensor_values(
                next_obs, next_actions_temp,
                network=self.qfs).max(2)[0].min(0)[0]

        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values

        q_target_detach = q_target.unsqueeze(0)
        # Use `expand` instead of `repeat` to avoid copying data
        q_target_detach = q_target_detach.expand(self.num_qs, -1, -1).contiguous()
        qfs_loss = self.qf_criterion(qs_pred, q_target_detach)
        qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()
        qfs_loss_total = qfs_loss
        if self.q_smooth_eps > 0 and self.q_smooth_reg > 0:
            M, A, size, noised_obs = self._get_noised_obs(obs, actions, self.q_smooth_eps)
            # (M,B*size,1)
            noised_qs_pred = self.qfs(noised_obs, actions.repeat_interleave(size, dim=0))
            diff = noised_qs_pred - qs_pred.repeat(1, 1, size).view(self.num_qs, -1, 1)     
            pos = torch.clamp(diff, min=0)
            neg = torch.clamp(diff, max=0)
            noise_Q_loss = (1-self.q_smooth_tau) *  pos.square().mean(axis=0) + self.q_smooth_tau * neg.square().mean(axis=0)
            noise_Q_loss = noise_Q_loss.view(M, size)
            noise_Q_loss_max = torch.mean(torch.max(noise_Q_loss,dim=1)[0])  
            qfs_loss_total += self.q_smooth_reg * noise_Q_loss_max
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q Smooth Loss'] = ptu.get_numpy(noise_Q_loss_max) * self.q_smooth_reg
                
        if self.q_ood_reg > 0: # self.q_ood_eps = 0 for PBRL
            ood_loss = torch.zeros(1, device=ptu.device)[0]
            if self.q_ood_uncertainty_reg > 0:
                M, A, size, noised_obs = self._get_noised_obs(obs, actions, self.q_ood_eps)
                ood_actions, _, _, _, *_ = self.policy(noised_obs, reparameterize=False)
                ood_qs_pred = self.qfs(noised_obs, ood_actions)
                ood_target = ood_qs_pred - self.q_ood_uncertainty_reg * ood_qs_pred.std(axis=0)
                ood_loss = self.qf_criterion(ood_target.detach(), ood_qs_pred).mean()
                qfs_loss_total += self.q_ood_reg * ood_loss

            if self.q_ood_uncertainty_reg > 0:
                self.q_ood_uncertainty_reg = max(self.q_ood_uncertainty_reg - self.q_ood_uncertainty_decay, self.q_ood_uncertainty_reg_min)
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q OOD Loss'] = ptu.get_numpy(ood_loss) * self.q_ood_reg
                self.eval_statistics['q_ood_uncertainty_reg'] = self.q_ood_uncertainty_reg
        
        if self.eta > 0:
            qs_pred_grads = None
            sample_size = min(qs_pred.size(0), actions.size(1))
            indices = np.random.choice(qs_pred.size(0), size=sample_size, replace=False)
            indices = torch.from_numpy(indices).long().to(ptu.device)

            obs_tile = obs.unsqueeze(0).expand(self.num_qs, -1, -1).contiguous()
            actions_tile = actions.unsqueeze(0).expand(self.num_qs, -1, -1).contiguous().requires_grad_(True)
            qs_preds_tile = self.qfs(obs_tile, actions_tile)
            qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)

            qs_pred_grads = torch.index_select(qs_pred_grads, dim=0, index=indices).transpose(0, 1)
            
            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            masks = torch.eye(sample_size, device=ptu.device).unsqueeze(dim=0).expand(qs_pred_grads.size(0), -1, -1).contiguous()
            qs_pred_grads = (1 - masks) * qs_pred_grads
            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (sample_size - 1)
            
            qfs_loss_total += self.eta * grad_loss

        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()

        self.try_update_target_networks()
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = ptu.get_numpy(log_pi - q_new_actions).mean()
            policy_avg_std = ptu.get_numpy(torch.exp(policy_log_std)).mean()
            self.eval_statistics['QFs Loss'] = np.mean(
                ptu.get_numpy(qfs_loss)) / self.num_qs
            if self.eta > 0:
                self.eval_statistics['Q Grad Loss'] = np.mean(
                    ptu.get_numpy(grad_loss))
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss)

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Predictions',
                    ptu.get_numpy(qs_pred),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Targets',
                    ptu.get_numpy(q_target),
                ))

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            self.eval_statistics['Policy std'] = np.mean(policy_avg_std)

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs, self.target_qfs,
                                self.soft_target_tau)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.qfs,
            log_alpha=self.log_alpha,
            policy_optim=self.policy_optimizer,
            qfs_optim=self.qfs_optimizer,
            alpha_optim=self.alpha_optimizer,
        )
    
    def load_snapshot(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.policy.load_state_dict(datas['trainer/policy'].state_dict())
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        self.target_qfs.load_state_dict(datas['trainer/target_qfs'].state_dict())
        self.log_alpha = datas['trainer/log_alpha']
        self.policy_optimizer.load_state_dict( datas['trainer/policy_optim'].state_dict())
        self.qfs_optimizer.load_state_dict(datas['trainer/qfs_optim'].state_dict())
        self.alpha_optimizer.load_state_dict(datas['trainer/alpha_optim'].state_dict())
        logger.log('Loading model from {} finished'.format(path))
    
    def load_qfs(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        logger.log('Loading Q networks from {} finished'.format(path))



class SACTrainerBatchEnsemble(TorchTrainer):
    """
    Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
    Continuous maximum Q-learning algorithm with parameterized actor.
    """
    def __init__(
            self,
            env,  # Associated environment for learning
            policy,  # Associated policy (should be TanhGaussian)
            qfs,  # Q functions
            target_qfs,  # Slow updater to Q functions
            discount=0.99,  # Discount factor
            reward_scale=1.0,  # Scaling of rewards to modulate entropy bonus
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,  # Target entropy for entropy-constraint variant
            policy_lr=3e-4,  # Learning rate of policy and entropy weight
            qf_lr=3e-4,  # Learning rate of Q functions
            optimizer_class=optim.Adam,  # Class of optimizer for all networks
            soft_target_tau=5e-3,  # Rate of update of target networks
            target_update_period=1,  # How often to update target networks
            max_q_backup=False,
            deterministic_backup=False,
            policy_eval_start=0,
            eta=-1.0,
            num_qs=10,
            replay_buffer=None,
            imagination=None,

            # smoothing
            num_samples=20,
            policy_smooth_eps=0,
            policy_smooth_reg=0.0,
            q_smooth_eps=0,
            q_smooth_reg=0.0,
            q_smooth_tau=0.2,
            norm_input=False,
            obs_std=1,
            q_ood_eps=0,
            q_ood_reg = 0,
            q_ood_uncertainty_reg=0,
            q_ood_uncertainty_reg_min=0,
            q_ood_uncertainty_decay=1e-6,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.num_qs = num_qs

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.eta = eta
        self.replay_buffer = replay_buffer

        #### robust 
        self.num_samples = num_samples
        self.policy_smooth_eps = policy_smooth_eps
        self.policy_smooth_reg = policy_smooth_reg
        self.q_smooth_eps = q_smooth_eps
        self.q_smooth_reg = q_smooth_reg
        self.q_smooth_tau = q_smooth_tau
        self.obs_std = 1 if not norm_input else ptu.from_numpy(obs_std)
        self.q_ood_eps = q_ood_eps
        self.q_ood_reg = q_ood_reg
        self.q_ood_uncertainty_reg = q_ood_uncertainty_reg
        self.q_ood_uncertainty_reg_min = q_ood_uncertainty_reg_min
        self.q_ood_uncertainty_decay = q_ood_uncertainty_decay

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic value: dimension of action space
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
            weight_decay=1e-3,
        )

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(-1, obs.shape[0], num_repeat, 1)
        return preds
    

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = action_shape // obs_shape
        obs_temp = obs.repeat_interleave(num_repeat, dim=0)
        preds = network(obs_temp, actions)
        preds = preds.view(-1, obs_shape, num_repeat, 1)

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions,
                                           1).view(obs.shape[0] * num_actions,
                                                   obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp,
            reparameterize=True,
            return_log_prob=True,
        )
        return new_obs_actions.detach(), new_obs_log_pi.view(
            obs.shape[0], num_actions, 1).detach()
        

    def _get_noised_obs(self, obs, actions, eps):
        """Return noise observation size BatchSize*Num_sample*ObservationSize
        """
        M, N = obs.shape[0], obs.shape[1] 
        A = actions.shape[1]
        size = self.num_samples
        delta_s = 2 * eps * self.obs_std * (torch.rand(size, N, device=ptu.device) - 0.5) 
        tmp_obs = obs.repeat_interleave(size,0).view(M,size,N)  
        noised_obs = tmp_obs + delta_s
        return M, A, size, noised_obs.view(-1,N)


    def train_from_torch(self, batch, indices):
        obs= batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        
        if self.eta > 0:
            actions.requires_grad_(True)
        
        """
        Policy and Alpha Loss
        """

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        # get min Q value
        q_new_actions,_,_ = self.qfs.sample(obs, new_obs_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._num_train_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.get_log_probs(obs.detach(), actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        
        if self.policy_smooth_eps > 0 and self.policy_smooth_reg > 0:
            M, A, size, noised_obs= self._get_noised_obs(obs, actions, self.policy_smooth_eps)
            _, noised_policy_mean, noised_policy_log_std, _, *_ = self.policy(noised_obs,reparameterize=True)
            action_dist = torch.distributions.Normal(policy_mean.repeat_interleave(size, dim=0),
                                                    policy_log_std.exp().repeat_interleave(size, dim=0))
                                                    
            noised_action_dist = torch.distributions.Normal(noised_policy_mean, noised_policy_log_std.exp())
            kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) + kl_divergence(noised_action_dist, action_dist).sum(axis=-1)
            kl_loss = kl_loss.view(M, size)
            max_values, _ = torch.max(kl_loss, dim=1)
            kl_loss_max = torch.mean(max_values)
            

            # noised_states_selected = noised_obs[np.arange(M), max_id]
            policy_loss += self.policy_smooth_reg * kl_loss_max
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Policy Smooth Loss'] = ptu.get_numpy(kl_loss_max) * self.policy_smooth_reg

        """
        QF Loss
        """
        # (num_qs, batch_size, output_size) ==> (batch_size, output_size)
        qs_pred = self.qfs(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs,
            reparameterize=False,
            return_log_prob=True,
        )

        if not self.max_q_backup:
            # Get min Q value
            target_q_values, _,_ = self.target_qfs.sample(next_obs, new_next_actions)
            if not self.deterministic_backup:
                target_q_values -= alpha * new_log_pi
        else:
            raise NotImplementedError
            # if self.max_q_backup
            # next_actions_temp, _ = self._get_policy_actions(
            #     next_obs, num_actions=10, network=self.policy)
            # target_q_values = self._get_tensor_values(
            #     next_obs, next_actions_temp,
            #     network=self.qfs).max(2)[0].min(0)[0]

        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values

        # (batch_size, output_size)
        q_target_detach = q_target.detach()
        qfs_loss = self.qf_criterion(qs_pred, q_target_detach)
        qfs_loss = qfs_loss.mean().sum()
        qfs_loss_total = qfs_loss

        if self.q_smooth_eps > 0 and self.q_smooth_reg > 0:
            # M is original shape
            M, A, size, noised_obs = self._get_noised_obs(obs, actions, self.q_smooth_eps)
            #  (batch_size*sample, output_size) Change to repeat_interleave
            noised_qs_pred = self.qfs(noised_obs, actions.repeat_interleave(size, dim=0))
            diff = (noised_qs_pred.view(M,size,-1) - qs_pred.view(M,1,-1)).view(M,size)            
            pos = torch.clamp(diff, min=0)
            neg = torch.clamp(diff, max=0)
            noise_Q_loss = (1-self.q_smooth_tau) * pos.square() + self.q_smooth_tau * neg.square()
            noise_Q_loss_max = torch.mean(torch.max(noise_Q_loss,dim=1)[0])  
            qfs_loss_total += self.q_smooth_reg * noise_Q_loss_max

            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q Smooth Loss'] = ptu.get_numpy(noise_Q_loss_max) * self.q_smooth_reg
                
        if self.q_ood_reg > 0: # self.q_ood_eps = 0 for PBRL
            ood_loss = torch.zeros(1, device=ptu.device)[0]
            if self.q_ood_uncertainty_reg > 0:
                M, A, size, noised_obs = self._get_noised_obs(obs, actions, self.q_ood_eps)
                ood_actions, _, _, _, *_ = self.policy(noised_obs, reparameterize=False)
                ood_qs_min,ood_qs_pred,ood_qs_std = self.qfs.sample(noised_obs, ood_actions)
                # THANH: 
                ood_target = ood_qs_pred - self.q_ood_uncertainty_reg * ood_qs_std
                ood_loss = self.qf_criterion(ood_target.detach(), ood_qs_pred).mean()
                qfs_loss_total += self.q_ood_reg * ood_loss

            if self.q_ood_uncertainty_reg > 0:
                self.q_ood_uncertainty_reg = max(self.q_ood_uncertainty_reg - self.q_ood_uncertainty_decay, self.q_ood_uncertainty_reg_min)
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q OOD Loss'] = ptu.get_numpy(ood_loss) * self.q_ood_reg
                self.eval_statistics['q_ood_uncertainty_reg'] = self.q_ood_uncertainty_reg
        
        if self.eta > 0:
            raise NotImplementedError
            # qs_pred_grads = None
            # sample_size = min(qs_pred.size(0), actions.size(1))
            # indices = np.random.choice(qs_pred.size(0), size=sample_size, replace=False)
            # indices = torch.from_numpy(indices).long().to(ptu.device)

            # obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
            # actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
            # qs_preds_tile = self.qfs(obs_tile, actions_tile)
            # qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            # qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)

            # qs_pred_grads = torch.index_select(qs_pred_grads, dim=0, index=indices).transpose(0, 1)
            
            # qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            # masks = torch.eye(sample_size, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            # qs_pred_grads = (1 - masks) * qs_pred_grads
            # grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (sample_size - 1)
            
            # qfs_loss_total += self.eta * grad_loss

        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()

        self.try_update_target_networks()
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = ptu.get_numpy(log_pi - q_new_actions).mean()
            policy_avg_std = ptu.get_numpy(torch.exp(policy_log_std)).mean()
            self.eval_statistics['QFs Loss'] = np.mean(
                ptu.get_numpy(qfs_loss)) / self.num_qs
            if self.eta > 0:
                raise NotImplementedError
                #self.eval_statistics['Q Grad Loss'] = np.mean(ptu.get_numpy(grad_loss))
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss)

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Predictions',
                    ptu.get_numpy(qs_pred),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Targets',
                    ptu.get_numpy(q_target),
                ))

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            self.eval_statistics['Policy std'] = np.mean(policy_avg_std)

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs, self.target_qfs,
                                self.soft_target_tau)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.qfs,
            log_alpha=self.log_alpha,
            policy_optim=self.policy_optimizer,
            qfs_optim=self.qfs_optimizer,
            alpha_optim=self.alpha_optimizer,
        )
    
    def load_snapshot(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.policy.load_state_dict(datas['trainer/policy'].state_dict())
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        self.target_qfs.load_state_dict(datas['trainer/target_qfs'].state_dict())
        self.log_alpha = datas['trainer/log_alpha']
        self.policy_optimizer.load_state_dict( datas['trainer/policy_optim'].state_dict())
        self.qfs_optimizer.load_state_dict(datas['trainer/qfs_optim'].state_dict())
        self.alpha_optimizer.load_state_dict(datas['trainer/alpha_optim'].state_dict())
        logger.log('Loading model from {} finished'.format(path))
    
    def load_qfs(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        logger.log('Loading Q networks from {} finished'.format(path))


class SACTrainerMimo(TorchTrainer):
    """
    Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
    Continuous maximum Q-learning algorithm with parameterized actor.
    """
    def __init__(
            self,
            env,  # Associated environment for learning
            policy,  # Associated policy (should be TanhGaussian)
            qfs,  # Q functions
            target_qfs,  # Slow updater to Q functions
            discount=0.99,  # Discount factor
            reward_scale=1.0,  # Scaling of rewards to modulate entropy bonus
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,  # Target entropy for entropy-constraint variant
            policy_lr=3e-4,  # Learning rate of policy and entropy weight
            qf_lr=3e-4,  # Learning rate of Q functions
            optimizer_class=optim.Adam,  # Class of optimizer for all networks
            soft_target_tau=5e-3,  # Rate of update of target networks
            target_update_period=1,  # How often to update target networks
            max_q_backup=False,
            deterministic_backup=False,
            policy_eval_start=0,
            eta=-1.0,
            num_qs=10,
            replay_buffer=None,
            imagination=None,

            # smoothing
            num_samples=20,
            policy_smooth_eps=0,
            policy_smooth_reg=0.0,
            q_smooth_eps=0,
            q_smooth_reg=0.0,
            q_smooth_tau=0.2,
            norm_input=False,
            obs_std=1,
            q_ood_eps=0,
            q_ood_reg = 0,
            q_ood_uncertainty_reg=0,
            q_ood_uncertainty_reg_min=0,
            q_ood_uncertainty_decay=1e-6,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.num_qs = num_qs

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.eta = eta
        self.replay_buffer = replay_buffer

        #### robust 
        self.num_samples = num_samples
        self.policy_smooth_eps = policy_smooth_eps
        self.policy_smooth_reg = policy_smooth_reg
        self.q_smooth_eps = q_smooth_eps
        self.q_smooth_reg = q_smooth_reg
        self.q_smooth_tau = q_smooth_tau
        self.obs_std = 1 if not norm_input else ptu.from_numpy(obs_std)
        self.q_ood_eps = q_ood_eps
        self.q_ood_reg = q_ood_reg
        self.q_ood_uncertainty_reg = q_ood_uncertainty_reg
        self.q_ood_uncertainty_reg_min = q_ood_uncertainty_reg_min
        self.q_ood_uncertainty_decay = q_ood_uncertainty_decay

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic value: dimension of action space
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(-1, obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions,
                                           1).view(obs.shape[0] * num_actions,
                                                   obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp,
            reparameterize=True,
            return_log_prob=True,
        )
        return new_obs_actions.detach(), new_obs_log_pi.view(
            obs.shape[0], num_actions, 1).detach()
        

    def _get_noised_obs(self, obs, actions, eps):
        M, N, A = obs.shape[0], obs.shape[1], actions.shape[1]
        size = self.num_samples
        delta_s = 2 * eps * self.obs_std * (torch.rand(size, N, device=ptu.device) - 0.5) 
        tmp_obs = obs.reshape(-1, 1, N).repeat(1, size, 1).reshape(-1, N)
        delta_s = delta_s.reshape(1, size, N).repeat(M, 1, 1).reshape(-1, N)
        noised_obs = tmp_obs + delta_s
        return M, A, size, noised_obs, delta_s


    def train_from_torch(self, batch, indices):
        obs= batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        
        if self.eta > 0:
            actions.requires_grad_(True)
        
        """
        Policy and Alpha Loss
        """

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        # get min Q value
        q_new_actions,_,_ = self.qfs.sample(obs, new_obs_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._num_train_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.get_log_probs(obs.detach(), actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        
        if self.policy_smooth_eps > 0 and self.policy_smooth_reg > 0:
            M, A, size, noised_obs, delta_s = self._get_noised_obs(obs, actions, self.policy_smooth_eps)
            _, noised_policy_mean, noised_policy_log_std, _, *_ = self.policy(noised_obs,reparameterize=True)
            action_dist = torch.distributions.Normal(policy_mean.reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A), policy_log_std.exp().reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A))
            noised_action_dist = torch.distributions.Normal(noised_policy_mean, noised_policy_log_std.exp())
            kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) + kl_divergence(noised_action_dist, action_dist).sum(axis=-1)
            kl_loss = kl_loss.reshape(M, size)
            max_id = torch.argmax(kl_loss, axis=1)
            kl_loss_max = kl_loss[np.arange(M), max_id].mean()
            # noised_states_selected = noised_obs[np.arange(M), max_id]
            policy_loss += self.policy_smooth_reg * kl_loss_max
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Policy Smooth Loss'] = ptu.get_numpy(kl_loss_max) * self.policy_smooth_reg

        """
        QF Loss
        """
        # (num_qs, batch_size, output_size) ==> (batch_size, output_size)
        qs_pred = self.qfs(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs,
            reparameterize=False,
            return_log_prob=True,
        )

        if not self.max_q_backup:
            # Get min Q value
            target_q_values, _,_ = self.target_qfs.sample(next_obs, new_next_actions)
            if not self.deterministic_backup:
                target_q_values -= alpha * new_log_pi
        else:
            raise NotImplementedError
            # if self.max_q_backup
            # next_actions_temp, _ = self._get_policy_actions(
            #     next_obs, num_actions=10, network=self.policy)
            # target_q_values = self._get_tensor_values(
            #     next_obs, next_actions_temp,
            #     network=self.qfs).max(2)[0].min(0)[0]

        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values

        # (batch_size, output_size)
        q_target_detach = q_target.detach()
        qfs_loss = self.qf_criterion(qs_pred, q_target_detach)
        qfs_loss = qfs_loss.mean().sum()
        qfs_loss_total = qfs_loss

        if self.q_smooth_eps > 0 and self.q_smooth_reg > 0:
            M, A, size, noised_obs, delta_s = self._get_noised_obs(obs, actions, self.q_smooth_eps)
            # (batch_size*sample, output_size) Change to repeat_interleave
            noised_qs_pred = self.qfs(noised_obs, actions.repeat_interleave(size, dim=0))
            diff = noised_qs_pred - qs_pred.repeat_interleave(size, dim=0)
            zero_tensor = torch.zeros(diff.shape, device=ptu.device)
            pos, neg = torch.maximum(diff, zero_tensor), torch.minimum(diff, zero_tensor)
            noise_Q_loss = (1-self.q_smooth_tau) *  pos.square() + self.q_smooth_tau * neg.square()
            noise_Q_loss = noise_Q_loss.reshape(size,M).T
            noise_Q_loss_max = noise_Q_loss[np.arange(M), torch.argmax(noise_Q_loss, axis=-1)].mean()
            qfs_loss_total += self.q_smooth_reg * noise_Q_loss_max
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q Smooth Loss'] = ptu.get_numpy(noise_Q_loss_max) * self.q_smooth_reg
                
        if self.q_ood_reg > 0: # self.q_ood_eps = 0 for PBRL
            ood_loss = torch.zeros(1, device=ptu.device)[0]
            if self.q_ood_uncertainty_reg > 0:
                M, A, size, noised_obs, delta_s = self._get_noised_obs(obs, actions, self.q_ood_eps)
                ood_actions, _, _, _, *_ = self.policy(noised_obs, reparameterize=False)
                ood_qs_min,ood_qs_pred,ood_qs_std = self.qfs.sample(noised_obs, ood_actions)
                # THANH: 
                ood_target = ood_qs_pred - self.q_ood_uncertainty_reg * ood_qs_std
                ood_loss = self.qf_criterion(ood_target.detach(), ood_qs_pred).mean()
                qfs_loss_total += self.q_ood_reg * ood_loss

            if self.q_ood_uncertainty_reg > 0:
                self.q_ood_uncertainty_reg = max(self.q_ood_uncertainty_reg - self.q_ood_uncertainty_decay, self.q_ood_uncertainty_reg_min)
            if self._need_to_update_eval_statistics:
                self.eval_statistics['Q OOD Loss'] = ptu.get_numpy(ood_loss) * self.q_ood_reg
                self.eval_statistics['q_ood_uncertainty_reg'] = self.q_ood_uncertainty_reg
        
        if self.eta > 0:
            raise NotImplementedError
            # qs_pred_grads = None
            # sample_size = min(qs_pred.size(0), actions.size(1))
            # indices = np.random.choice(qs_pred.size(0), size=sample_size, replace=False)
            # indices = torch.from_numpy(indices).long().to(ptu.device)

            # obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
            # actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
            # qs_preds_tile = self.qfs(obs_tile, actions_tile)
            # qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            # qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)

            # qs_pred_grads = torch.index_select(qs_pred_grads, dim=0, index=indices).transpose(0, 1)
            
            # qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            # masks = torch.eye(sample_size, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            # qs_pred_grads = (1 - masks) * qs_pred_grads
            # grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (sample_size - 1)
            
            # qfs_loss_total += self.eta * grad_loss

        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()

        self.try_update_target_networks()
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = ptu.get_numpy(log_pi - q_new_actions).mean()
            policy_avg_std = ptu.get_numpy(torch.exp(policy_log_std)).mean()
            self.eval_statistics['QFs Loss'] = np.mean(
                ptu.get_numpy(qfs_loss)) / self.num_qs
            if self.eta > 0:
                raise NotImplementedError
                #self.eval_statistics['Q Grad Loss'] = np.mean(ptu.get_numpy(grad_loss))
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss)

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Predictions',
                    ptu.get_numpy(qs_pred),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Targets',
                    ptu.get_numpy(q_target),
                ))

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            self.eval_statistics['Policy std'] = np.mean(policy_avg_std)

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs, self.target_qfs,
                                self.soft_target_tau)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.qfs,
            log_alpha=self.log_alpha,
            policy_optim=self.policy_optimizer,
            qfs_optim=self.qfs_optimizer,
            alpha_optim=self.alpha_optimizer,
        )
    
    def load_snapshot(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.policy.load_state_dict(datas['trainer/policy'].state_dict())
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        self.target_qfs.load_state_dict(datas['trainer/target_qfs'].state_dict())
        self.log_alpha = datas['trainer/log_alpha']
        self.policy_optimizer.load_state_dict( datas['trainer/policy_optim'].state_dict())
        self.qfs_optimizer.load_state_dict(datas['trainer/qfs_optim'].state_dict())
        self.alpha_optimizer.load_state_dict(datas['trainer/alpha_optim'].state_dict())
        logger.log('Loading model from {} finished'.format(path))
    
    def load_qfs(self, path):
        datas = torch.load(path, map_location=ptu.device)
        self.qfs.load_state_dict(datas['trainer/qfs'].state_dict())
        logger.log('Loading Q networks from {} finished'.format(path))