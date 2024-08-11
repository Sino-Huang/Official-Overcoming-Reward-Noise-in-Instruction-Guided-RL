#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=2

#SBATCH --time=1-04:00:00

#SBATCH --mem=64G

#SBATCH --partition=deeplearn

#SBATCH -A punim0478

#SBATCH --gres=gpu:1

#SBATCH -q gpgpudeeplearn

#COMMENTOUT #SBATCH --constraint=dlg3

export PYTHONPATH=`pwd`



# ! 4 train pure rl policy model GPU MEM ~= 7400MiB
# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=7738"

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=7738"

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=minigrid,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=7738"


# # ! 5 train pure rl + exploration rewards ~= 19300MiB

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565"

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=minigrid,purpose=policy,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565"

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565"


# # ! 4 train pure rl policy model GPU MEM ~= 7400MiB
# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565"

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=crafter,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565"

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=minigrid,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565"


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=7738,train_rl_policy_params.montezuma_env_params.int_rew_type=rnd" &

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=2565,train_rl_policy_params.montezuma_env_params.int_rew_type=rnd" &


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=false,general.seed=1234,train_rl_policy_params.montezuma_env_params.int_rew_type=rnd" &

wait