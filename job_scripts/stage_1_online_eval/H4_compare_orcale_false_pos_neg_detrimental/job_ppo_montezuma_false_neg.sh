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

kedro run --from-nodes setup_env_node --to-nodes train_rl_main_node --params env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H4_1_False_Negative_Sim,reward_machine_params.lang_reward_machine_type=false_negative_sim,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=2565,reward_machine_params.oracle_error_rate=0.1


kedro run --from-nodes setup_env_node --to-nodes train_rl_main_node --params env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H4_1_False_Negative_Sim,reward_machine_params.lang_reward_machine_type=false_negative_sim,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=2565,reward_machine_params.oracle_error_rate=0.2 

kedro run --from-nodes setup_env_node --to-nodes train_rl_main_node --params env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H4_1_False_Negative_Sim,reward_machine_params.lang_reward_machine_type=false_negative_sim,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=2565,reward_machine_params.oracle_error_rate=0.3 


kedro run --from-nodes setup_env_node --to-nodes train_rl_main_node --params env=montezuma,purpose=policy,train_rl_policy_params.eval_type_tag=H4_1_False_Negative_Sim,reward_machine_params.lang_reward_machine_type=false_negative_sim,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,general.seed=2565,reward_machine_params.oracle_error_rate=0.4


