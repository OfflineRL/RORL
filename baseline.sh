# bash baseline.sh 0  


# bash baseline.sh 0 halfcheetah-random-v2  0 
# bash baseline.sh 1 halfcheetah-expert-v2  1
# bash baseline.sh 0  halfcheetah-medium-v2  2
# bash baseline.sh 1 halfcheetah-medium-expert-v2 3
# bash baseline.sh 2 halfcheetah-medium-replay-v2 4

# bash baseline.sh 2 hopper-random-v2 5
# bash baseline.sh 3 hopper-expert-v2 6
# bash baseline.sh 3 hopper-medium-v2 7
# bash baseline.sh 4 hopper-medium-expert-v2 8
# bash baseline.sh 4 hopper-medium-replay-v2 9

# bash baseline.sh 5 walker2d-random-v2 10
# bash baseline.sh 5 walker2d-expert-v2 11
# bash baseline.sh 6 walker2d-medium-v2 12
# bash baseline.sh 6 walker2d-medium-expert-v2 13     
# bash baseline.sh 7 walker2d-medium-replay-v2 14



WANDB_RUN_GROUP=$3 CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name $2 --seed 0 \
    --norm_input --load_config_type benchmark --epoch 1000\
    --exp_prefix RORL_BASELINE_CONFIG_MIMIC_MEDIAN  --tensorboard --wandb

# WANDB_RUN_GROUP=$3 CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name $2 --seed 1 \
#     --norm_input --load_config_type benchmark --epoch 1000\
#     --exp_prefix RORL_BASELINE --tensorboard --wandb


# WANDB_RUN_GROUP=$3 CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name $2 --seed 2 \
#     --norm_input --load_config_type benchmark --epoch 1000\
#     --exp_prefix RORL_BASELINE --tensorboard --wandb