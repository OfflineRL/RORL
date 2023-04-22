CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name hopper-medium-expert-v2 \
    --norm_input --epoch 10 --num_qs 10\
    --exp_prefix RORL_GAUSS_HOPPER --tensorboard 
 


# WANDB_RUN_GROUP=$3 CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name $2 --seed 0 \
#     --norm_input --load_config_type benchmark --epoch 1000\
#     --exp_prefix RORL_GAUSS_HOPPER  --tensorboard --wandb




# CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name halfcheetah-expert-v2 \
#     --norm_input --load_config_type benchmark \
#     --exp_prefix RORL_OPTUNE --tuning --epoch 10
 