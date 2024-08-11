#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=2

#SBATCH --time=12-04:00:00

#SBATCH --mem=64G

#SBATCH --partition=deeplearn

#SBATCH -A punim0478

#SBATCH --gres=gpu:1

#SBATCH -q gpgpudeeplearn

#COMMENTOUT #SBATCH --constraint=dlg3

export PYTHONPATH=`pwd`


# traj length = 5 minigrid
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.traj_length=5,lang_rew_model_params.has_extra_data_manipulation=true"

# traj length = 5 crafter
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.traj_length=5,lang_rew_model_params.has_extra_data_manipulation=true"


# markovian minigrid pretrain
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,lang_rew_model_params.model_kwargs.minigrid_no_pretrain=true,lang_rew_model_params.algorithm_kwargs.lr=2.0e-4"

kedro run --from-nodes train_validate_test_split_node --to-nodes train_lang_rew_model_node --params env=minigrid,purpose=lang,lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,lang_rew_model_params.model_kwargs.minigrid_no_pretrain=true,lang_rew_model_params.algorithm_kwargs.lr=1.0e-4


# standard
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=crafter,purpose=lang"

# standard
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=montezuma,purpose=lang,lang_rew_model_params.algorithm_kwargs.continue_training=true"

# ! markovian
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=crafter,purpose=lang,lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false"

# ! markovian
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=montezuma,purpose=lang,lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false"

# ! xclip
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=crafter,purpose=lang,lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=8,lang_rew_model_params.has_extra_data_manipulation=true,lang_rew_model_params.model_kwargs.pretrained_model_cls=microsoft/xclip-base-patch16"

# ! xclip
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=montezuma,purpose=lang,lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=8,lang_rew_model_params.has_extra_data_manipulation=true,lang_rew_model_params.model_kwargs.pretrained_model_cls=microsoft/xclip-base-patch16"


