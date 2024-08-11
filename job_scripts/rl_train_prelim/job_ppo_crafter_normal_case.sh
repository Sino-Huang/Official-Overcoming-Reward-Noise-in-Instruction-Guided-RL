#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=2

#SBATCH --time=6-04:00:00

#SBATCH --mem=128G

#SBATCH --partition=deeplearn

#SBATCH -A punim0478

#SBATCH --gres=gpu:1

#SBATCH -q gpgpudeeplearn

#COMMENTOUT #SBATCH --constraint=dlg3

export PYTHONPATH=`pwd`

# pure ppo normal case crafter env

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=7738" &

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565" &

wait

sleep 6


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=1234" &


# ppo + int_rew normal case crafter env
kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=7738" &

wait 

sleep 6

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565" &


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=1234" &

wait

