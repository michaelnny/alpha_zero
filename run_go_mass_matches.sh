#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Try to find the best models by playing some matches against each other

# black_model=137000
# for (( white_model=130000; white_model<=160000; white_model+=1000 ))
# do
#     python3 -m alpha_zero.play.eval_agent_go_mass_matches --num_games=50 \
#         --num_simulations=200 \
#         --num_res_blocks=10 \
#         --num_filters=128 \
#         --num_fc_units=128 \
#         --save_match_dir=./9x9_matches/${black_model}_vs_${white_model} \
#         --black_ckpt=./checkpoints/go/9x9/training_steps_${black_model}.ckpt \
#         --white_ckpt=./checkpoints/go/9x9/training_steps_${white_model}.ckpt
# done


black_model=154000
white_models=( 151000 152000 153000 145000 160000 159000 149000 146000 150000 147000)
for white_model in "${white_models[@]}"
do
    python3 -m alpha_zero.play.eval_agent_go_mass_matches --num_games=50 \
        --num_simulations=200 \
        --num_res_blocks=10 \
        --num_filters=128 \
        --num_fc_units=128 \
        --save_match_dir=./9x9_matches/${black_model}_vs_${white_model} \
        --black_ckpt=./checkpoints/go/9x9/training_steps_${black_model}.ckpt \
        --white_ckpt=./checkpoints/go/9x9/training_steps_${white_model}.ckpt
done
