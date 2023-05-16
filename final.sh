# bash final.sh halfcheetah-medium-v2  final_halfcheetah_medium 4 0 
# bash final.sh halfcheetah-medium-expert-v2 final_halfcheetah_medium_expert 6 0.667
# bash final.sh halfcheetah-expert-v2 final_halfcheetah_expert 5 0.72
# bash final.sh halfcheetah-medium-replay-v2 final_halfcheetah_medium_replay 
# bash final.sh halfcheetah-random-v2 final_halfcheetah_random 1


# bash final.sh hopper-medium-v2  final_hopper_medium 1
# bash final.sh hopper-medium-expert-v2 final_hopper_medium_expert 11 0.243
# bash final.sh hopper-expert-v2 final_hopper_expert 1
# bash final.sh hopper-medium-replay-v2 final_hopper_medium_replay 1
# bash final.sh hopper-random-v2 final_hopper_random 1


# bash final.sh walker2d-medium-v2  final_walker2d_medium 1
# bash final.sh walker2d-medium-expert-v2 final_walker2d_medium_expert 10 2.4533
# bash final.sh walker2d-expert-v2 final_walker2d_expert 1
# bash final.sh walker2d-medium-replay-v2 final_walker2d_medium_replay 3 4.437
# bash final.sh walker2d-random-v2 final_walker2d_random 1

#ABSLATION
#bash final.sh walker2d-medium-expert-v2 final_walker2d_medium_expert 9 4.4459 

WANDB_RUN_GROUP=$2 CUDA_VISIBLE_DEVICES=$5 python -m scripts.sac -a RANKONE --env_name $1 --seed 0 --num_qs $3\
    --norm_input --epoch 3000 --exp_prefix REPORT_TUNE --tensorboard --wandb --policy_eval_start 0\
    --policy_smooth_reg 0.0 --policy_smooth_eps 0.001 \
    --q_smooth_reg 0.0 --q_smooth_eps 0.001 \
    --q_ood_reg $4 --q_ood_uncertainty_reg 1.5 --q_ood_eps 0.03     


# WANDB_RUN_GROUP=$2 CUDA_VISIBLE_DEVICES=$3 python -m scripts.sac -a RANKONE --env_name $1 --seed 1 --num_qs $3\
#     --norm_input --epoch 3000 --exp_prefix REPORT  --tensorboard --wandb --policy_eval_start 0\
#     --policy_smooth_reg 0.0 --policy_smooth_eps 0.001 \
#     --q_smooth_reg 0.0 --q_smooth_eps 0.001 \
#     --q_ood_reg $4 --q_ood_uncertainty_reg 1.5 --q_ood_eps 0.03     


# WANDB_RUN_GROUP=$2 CUDA_VISIBLE_DEVICES=$3 python -m scripts.sac -a RANKONE --env_name $1 --seed 2 --num_qs $3\
#     --norm_input --epoch 3000 --exp_prefix REPORT  --tensorboard --wandb --policy_eval_start 0\
#     --policy_smooth_reg 0.0 --policy_smooth_eps 0.001 \
#     --q_smooth_reg 0.0 --q_smooth_eps 0.001 \
#     --q_ood_reg $4 --q_ood_uncertainty_reg 1.5 --q_ood_eps 0.03     