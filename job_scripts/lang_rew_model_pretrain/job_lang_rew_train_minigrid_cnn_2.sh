#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=2

#SBATCH --time=12-04:00:00

#SBATCH --mem=128G

#SBATCH --partition=deeplearn

#SBATCH -A punim0478

#SBATCH --gres=gpu:1

#SBATCH -q gpgpudeeplearn

#COMMENTOUT #SBATCH --constraint=dlg3

export PYTHONPATH=`pwd`

# ! CNN markovian 
kedro run --from-nodes train_validate_test_split_node --to-nodes train_lang_rew_model_node --params "env=minigrid,purpose=lang,lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,lang_rew_model_params.model_kwargs.minigrid_no_pretrain=true,lang_rew_model_params.algorithm_kwargs.lr=5.0e-5,lang_rew_model_params.algorithm_kwargs.continue_training=true"  & 


# ! CNN standard 
kedro run --from-nodes "train_validate_test_split_node" --to-nodes "train_lang_rew_model_node" --params "env=minigrid,purpose=lang,lang_rew_model_params.model_kwargs.minigrid_no_pretrain=true,lang_rew_model_params.algorithm_kwargs.lr=5.0e-5,lang_rew_model_params.algorithm_kwargs.continue_training=true"   & 


wait 


