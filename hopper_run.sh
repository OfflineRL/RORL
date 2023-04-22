# CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name halfcheetah-medium-v2 --num_qs $3 \
#     --norm_input --load_config_type benchmark --epoch 1000\
#     --exp_prefix RORL_RANK1OPV1_BS$2_RELU --batch_size=$2

CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name hopper-medium-v2 \
    --norm_input --load_config_type benchmark \
    --exp_prefix RORL_OPTUNE_HOPPER --tensorboard
 
