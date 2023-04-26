# bash run.sh halfcheetah-medium-v2  halfcheetah_medium 1
WANDB_RUN_GROUP=$2 CUDA_VISIBLE_DEVICES=$3 python -m scripts.sac --env_name $1 --seed 0 \
    --norm_input --epoch 1000 --exp_prefix RORL_GAUSS_SMOOTH_QPI_NOOOD  --tensorboard --wandb \
    --policy_smooth_reg 0.1 --policy_smooth_eps 0.001 \
    --q_smooth_reg 0.1 --q_smooth_eps 0.001 \
    --q_ind_uncertainty_reg 1.4\
    --q_ood_reg 0.0 --q_ood_uncertainty_reg 1.5 --q_ood_eps 0.03     



# CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name hopper-medium-expert-v2 \
#     --norm_input --epoch 10 --num_qs 10\
#     --exp_prefix RORL_GAUSS_HOPPER --tensorboard 
 


# WANDB_RUN_GROUP=$3 CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name $2 --seed 0 \
#     --norm_input --load_config_type benchmark --epoch 1000\
#     --exp_prefix RORL_GAUSS_HOPPER  --tensorboard --wandb




# CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name halfcheetah-expert-v2 \
#     --norm_input --load_config_type benchmark \
#     --exp_prefix RORL_OPTUNE --tuning --epoch 10
 