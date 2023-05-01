 


# bash tune.sh halfcheetah-random-v2  0 
# bash tune.sh halfcheetah-expert-v2  1
# bash tune.sh  halfcheetah-medium-v2  2
# bash tune.sh halfcheetah-medium-expert-v2 3
# bash tune.sh halfcheetah-medium-replay-v2 4

# bash tune.sh hopper-random-v2 hopper_random 5
# bash tune.sh hopper-expert-v2 hopper_expert 
# bash tune.sh hopper-medium-v2 hopper_medium 7
# bash tune.sh hopper-medium-expert-v2 hopper_random_medium_expert 8
# bash tune.sh hopper-medium-replay-v2 hopper_random_medium_replay 9

# bash tune.sh walker2d-random-v2 10
# bash tune.sh walker2d-expert-v2 11
# bash tune.sh walker2d-medium-v2 12
# bash tune.sh walker2d-medium-expert-v2 13     
# bash tune.sh walker2d-medium-replay-v2 14

# NOW: bash tune.sh hopper-medium-expert-v2 hopper_medium_expert 1
# bash tune.sh halfcheetah-medium-v2  halfcheetah_medium 1

GTIMER_DISABLE=1 WANDB_RUN_GROUP=$2 CUDA_VISIBLE_DEVICES=$3 python -m scripts.sac --env_name $1 --seed 0 \
    --norm_input --epoch 1000 --exp_prefix TUNE_MIN_BC_SAC --tensorboard --wandb \
    --policy_smooth_eps 0.000 --q_smooth_eps 0.000 \
    --policy_smooth_reg 0.0 --policy_smooth_eps 0.001 \
    --q_smooth_reg 0.0 --q_smooth_eps 0.001 \
    --q_ind_uncertainty_reg 0.0\
    --q_ood_reg 10 --q_ood_uncertainty_reg 0.0 --q_ood_eps 0.03 \
    --tuning     


# WANDB_RUN_GROUP=$2 CUDA_VISIBLE_DEVICES=$3 python -m scripts.sac --env_name $1 --seed 0 \
#     --norm_input --epoch 1000 --exp_prefix RORL_TUNE_GAUSS_TEST  --tensorboard --wandb \
#     --policy_smooth_eps 0.005 --q_smooth_eps 0.005 \
#     --q_ood_uncertainty_reg 1 --q_ood_eps 0.01 --q_ood_reg 0.5 #--tuning 
 