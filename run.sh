# bash run.sh halfcheetah-medium-v2  halfcheetah_medium 1
# bash run.sh halfcheetah-medium-expert-v2 halfcheetah-medium-expert 1
# bash run.sh halfcheetah-expert-v2 halfcheetah-expert 1
# bash run.sh halfcheetah-medium-replay-v2 halfcheetah-medium-replay 1
# bash run.sh halfcheetah-random-v2 halfcheetah-random 1


# bash run.sh hopper-medium-v2  hopper_medium 1
# bash run.sh hopper-medium-expert-v2 hopper-medium-expert 1
# bash run.sh hopper-expert-v2 hopper-expert 1
# bash run.sh hopper-medium-replay-v2 hopper-medium-replay 1
# bash run.sh hopper-random-v2 hopper-random 1


# bash run.sh walker2d-medium-v2  walker2d-medium 1
# bash run.sh walker2d-medium-expert-v2 walker2d-medium-expert 1
# bash run.sh walker2d-expert-v2 walker2d-expert 1
# bash run.sh walker2d-medium-replay-v2 walker2d-medium-replay 1
# bash run.sh walker2d-random-v2 walker2d-random 1


WANDB_RUN_GROUP=$2 CUDA_VISIBLE_DEVICES=$3 python -m scripts.sac -a RANKONE --env_name $1 --seed 0 \
    --norm_input --epoch 1000 --exp_prefix RANKONE_MIN  --tensorboard --wandb --policy_eval_start 0\
    --policy_smooth_reg 0.0 --policy_smooth_eps 0.001 \
    --q_smooth_reg 0.0 --q_smooth_eps 0.001 \
    --q_ind_uncertainty_reg 0.0\
    --q_ood_reg 0.9 --q_ood_uncertainty_reg 1.5 --q_ood_eps 0.03     

