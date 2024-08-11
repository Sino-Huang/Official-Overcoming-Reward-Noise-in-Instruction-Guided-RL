#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=2

#SBATCH --time=4-04:00:00

#SBATCH --mem=64G

#SBATCH --partition=deeplearn

#SBATCH -A punim0478

#SBATCH --gres=gpu:1

#SBATCH -q gpgpudeeplearn

#COMMENTOUT #SBATCH --constraint=dlg3

export PYTHONPATH=`pwd`

# pure ppo no_temporal_order_sim montezuma env

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H1_1_Compo_No_Temp_Order_Sim,reward_machine_params.lang_reward_machine_type=no_temporal_order_sim,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=7738" &

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H1_1_Compo_No_Temp_Order_Sim,reward_machine_params.lang_reward_machine_type=no_temporal_order_sim,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=2565" &


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H1_1_Compo_No_Temp_Order_Sim,reward_machine_params.lang_reward_machine_type=no_temporal_order_sim,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=1234" &

wait

# sleep 10 sec
sleep 10

# ppo + int_rew no_temporal_order_sim montezuma env
kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H1_1_Compo_No_Temp_Order_Sim,reward_machine_params.lang_reward_machine_type=no_temporal_order_sim,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=7738" &

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H1_1_Compo_No_Temp_Order_Sim,reward_machine_params.lang_reward_machine_type=no_temporal_order_sim,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=2565" &


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params "env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H1_1_Compo_No_Temp_Order_Sim,reward_machine_params.lang_reward_machine_type=no_temporal_order_sim,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=1234" &

wait