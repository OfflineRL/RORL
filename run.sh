CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name hopper-medium-expert-v2 \
    --norm_input --load_config_type benchmark --epoch 1000 --num_qs $2\
    --exp_prefix RORL_RANK1GAUSS_HOPPER --tensorboard 
 

# CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name halfcheetah-expert-v2 \
#     --norm_input --load_config_type benchmark \
#     --exp_prefix RORL_OPTUNE --tuning --epoch 10
 