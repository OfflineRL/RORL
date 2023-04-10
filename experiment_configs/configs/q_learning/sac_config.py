from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
from lifelong_rl.models import networks
from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.q_learning import sac
import lifelong_rl.util.pythonplusplus as ppp
import os
import torch
import lifelong_rl.torch.pytorch_util as ptu
from torch.nn import functional as F


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
        efficient="RANK1",
):
    """
    Policy construction
    """
    assert efficient in ["ENSAMBLE", "MIMO", "BATCH_ENSEMBLE","RANK1"]
    num_qs = variant['trainer_kwargs']['num_qs']
    M = variant['policy_kwargs']['layer_size']
    num_q_layers = variant['policy_kwargs']['num_q_layers']
    num_p_layers = variant['policy_kwargs']['num_p_layers']

    # normalization
    norm_input = variant['norm_input']
    obs_norm_mean, obs_norm_std = variant['normalization_info'][
        'obs_mean'], variant['normalization_info']['obs_std']

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_p_layers,
        layer_norm=None,
        norm_input=norm_input,
        obs_norm_mean=obs_norm_mean,
        obs_norm_std=obs_norm_std,
    )

    if efficient == "ENSAMBLE":
        qfs, target_qfs = ppp.group_init(
            2,
            ParallelizedEnsembleFlattenMLP,
            ensemble_size=num_qs,
            hidden_sizes=[M] * num_q_layers,
            input_size=obs_dim + action_dim,
            output_size=1,
            layer_norm=None,
            norm_input=norm_input,
            obs_norm_mean=obs_norm_mean,
            obs_norm_std=obs_norm_std,
        )

        trainer = sac.SACTrainer(
            env=eval_env,
            policy=policy,
            qfs=qfs,
            target_qfs=target_qfs,
            replay_buffer=replay_buffer,
            norm_input=norm_input,
            obs_std=obs_norm_std,
            **variant['trainer_kwargs'],
        )

    elif efficient == "RANK1":
        qfs, target_qfs = ppp.group_init(
            2,
            networks.BatchEnsembleFlattenRank1,
            ensemble_size=num_qs,
            hidden_sizes=[M] * num_q_layers,
            input_size=obs_dim + action_dim,
            output_size=1,
            layer_norm=None,
            norm_input=norm_input,
            obs_norm_mean=obs_norm_mean,
            obs_norm_std=obs_norm_std,
        )

        trainer = sac.SACTrainerRank1(
            env=eval_env,
            policy=policy,
            qfs=qfs,
            target_qfs=target_qfs,
            replay_buffer=replay_buffer,
            norm_input=norm_input,
            obs_std=obs_norm_std,
            **variant['trainer_kwargs'],
        )


    elif efficient == "MIMO":
        qfs, target_qfs = ppp.group_init(
            2,
            networks.BatchEnsembleFlattenMLP,
            ensemble_size=num_qs,
            hidden_sizes=[M] * num_q_layers,
            input_size=obs_dim + action_dim,
            output_size=1,
            layer_norm=None,
            norm_input=norm_input,
            obs_norm_mean=obs_norm_mean,
            obs_norm_std=obs_norm_std,
        )

        trainer = sac.SACTrainerMimo(
            env=eval_env,
            policy=policy,
            qfs=qfs,
            target_qfs=target_qfs,
            replay_buffer=replay_buffer,
            norm_input=norm_input,
            obs_std=obs_norm_std,
            **variant['trainer_kwargs'],
        )

    elif efficient == "BATCH_ENSEMBLE":
        qfs, target_qfs = ppp.group_init(
            2,
            networks.BatchEnsembleFlattenMLP,
            ensemble_size=num_qs,
            hidden_sizes=[M] * num_q_layers,
            input_size=obs_dim + action_dim,
            output_size=1,
            layer_norm=None,
            norm_input=norm_input,
            obs_norm_mean=obs_norm_mean,
            obs_norm_std=obs_norm_std,
        )

        trainer = sac.SACTrainerBatchEnsemble(
            env=eval_env,
            policy=policy,
            qfs=qfs,
            target_qfs=target_qfs,
            replay_buffer=replay_buffer,
            norm_input=norm_input,
            obs_std=obs_norm_std,
            **variant['trainer_kwargs'],
        )
    else:
        raise NotImplementedError

    if variant['load_path'] != '':
        trainer.load_snapshot(variant['load_path'])
    if variant['load_Qs'] != '':
        trainer.load_qfs(variant['load_Qs'])
    """
    Create config dict
    """

    config = dict()
    config.update(
        dict(
            trainer=trainer,
            exploration_policy=policy,
            evaluation_policy=MakeDeterministic(policy),
            exploration_env=expl_env,
            evaluation_env=eval_env,
            replay_buffer=replay_buffer,
            qfs=qfs,
        ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())

    return config
