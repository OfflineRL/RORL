# bash eval.sh halfcheetah-medium-v2  
# bash eval.sh halfcheetah-medium-expert-v2 
# bash eval.sh halfcheetah-expert-v2 
# bash eval.sh halfcheetah-medium-replay-v2 
# bash eval.sh halfcheetah-random-v2 


# bash eval.sh hopper-medium-v2  
# bash eval.sh hopper-medium-expert-v2 
# bash eval.sh hopper-expert-v2 
# bash eval.sh hopper-medium-replay-v2 
# bash eval.sh hopper-random-v2 


# bash eval.sh walker2d-medium-v2  
# bash eval.sh walker2d-medium-expert-v2 
# bash eval.sh walker2d-expert-v2 
# bash eval.sh walker2d-medium-replay-v2 
# bash eval.sh walker2d-random-v2 






python -m scripts.sac --env_name $1 --num_qs $3 --norm_input --eval_no_training --load_path $2 --exp_prefix eval_RANKONE_FINAL