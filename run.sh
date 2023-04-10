CUDA_VISIBLE_DEVICES=$1 python -m scripts.sac --env_name halfcheetah-medium-v2 --num_qs 8 \
    --norm_input --load_config_type benchmark \
    --exp_prefix RORL_BS$2 --batch_size=$2
