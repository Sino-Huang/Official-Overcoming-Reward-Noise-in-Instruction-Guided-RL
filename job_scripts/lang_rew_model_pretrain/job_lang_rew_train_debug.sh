#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=2

#SBATCH --time=4-04:00:00

#SBATCH --mem=64G

#SBATCH --partition=feit-gpu-a100

#SBATCH --qos=feit

#SBATCH --gres=gpu:1


export PYTHONPATH=`pwd`

# ! standard but train with CNN smaller lr
kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.model_kwargs.minigrid_no_pretrain=true,lang_rew_model_params.algorithm_kwargs.lr=1.0e-4"


# ! standard but train with CNN 
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.model_kwargs.minigrid_no_pretrain=true"

# ! xclip
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.is_markovian=false,lang_rew_model_params.traj_length=8,lang_rew_model_params.has_extra_data_manipulation=true,lang_rew_model_params.model_kwargs.pretrained_model_cls=microsoft/xclip-base-patch16"

# ! markovian
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false"

# ! standard 
# kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang"