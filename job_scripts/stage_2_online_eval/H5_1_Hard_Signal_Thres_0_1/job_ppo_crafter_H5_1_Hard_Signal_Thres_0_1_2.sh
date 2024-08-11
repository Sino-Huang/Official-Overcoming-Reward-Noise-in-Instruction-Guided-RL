#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=4

#SBATCH --time=12-04:00:00

#SBATCH --mem=64G

#SBATCH --partition=deeplearn

#SBATCH -A punim0478

#SBATCH --gres=gpu:1

#SBATCH -q gpgpudeeplearn

#COMMENTOUT #SBATCH --constraint=dlg3

export PYTHONPATH=`pwd`

# pure ppo standard crafter env traj_length=2

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,\
# general.seed=7738  &


# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,\
# general.seed=2565  &



# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,\
# general.seed=1234  &


# wait

# # sleep 10 sec
# sleep 10

# # pure ppo standard crafter env traj_length=10 

kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=10,lang_rew_model_params.has_extra_data_manipulation=true,\
general.seed=7738  &


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=10,lang_rew_model_params.has_extra_data_manipulation=true,\
general.seed=2565  &


kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=10,lang_rew_model_params.has_extra_data_manipulation=true,\
general.seed=1234  &

wait 
sleep 10 



# # ppo + int_rew standard crafter env traj_length=2
# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,\
# general.seed=7738  &


# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,\
# general.seed=2565  &



# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,\
# general.seed=1234  &


# wait

# # sleep 10 sec
# sleep 10

# # ppo + int_rew standard crafter env traj_length=10
# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=10,lang_rew_model_params.has_extra_data_manipulation=true,\
# general.seed=7738  &


# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=10,lang_rew_model_params.has_extra_data_manipulation=true,\
# general.seed=2565  &


# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H5_1_Hard_Signal_Thres_0_1,reward_machine_params.lang_reward_machine_type=standard,rl_policy_setup_params.is_int_rew_activated=true,rl_policy_setup_params.is_lang_rew_activated=true,rl_policy_setup_params.lang_rew_settings.has_hard_signal=true,rl_policy_setup_params.lang_rew_settings.hard_signal_cp_error_rate="0.1",\
# lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=10,lang_rew_model_params.has_extra_data_manipulation=true,\
# general.seed=1234  &

# wait 