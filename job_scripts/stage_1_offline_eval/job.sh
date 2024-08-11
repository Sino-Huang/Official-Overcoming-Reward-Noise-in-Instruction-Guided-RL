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

kedro run --from-nodes train_validate_test_split_node,setup_minigrid_generalization_testing_dataloader_node --to-nodes "eval_lang_rew_model_node" --params env=montezuma,purpose=lang,eval_lrm_params.is_offline_eval_completed=true

kedro run --from-nodes train_validate_test_split_node,setup_minigrid_generalization_testing_dataloader_node --to-nodes "eval_lang_rew_model_node" --params env=minigrid,purpose=lang

kedro run --from-nodes train_validate_test_split_node,setup_minigrid_generalization_testing_dataloader_node --to-nodes "eval_lang_rew_model_node" --params env=crafter,purpose=lang