"""
This is a boilerplate pipeline 'minigrid_diff_env_testing_dataloader'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import setup_minigrid_generalization_testing_dataloader

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=setup_minigrid_generalization_testing_dataloader,
                inputs=[
                    "minigrid_smaller_env_expert_traj_data#pkl",
                    "minigrid_smaller_env_expert_instr_data#csv",
                    "minigrid_larger_env_expert_traj_data#pkl",
                    "minigrid_larger_env_expert_instr_data#csv",
                    "params:minigrid_diff_env_testing_dataloader_params",
                    "params:lang_rew_model_params",
                    "params:traj_instr_pairs_params",
                    "params:general",
                ],
                outputs=["smaller_env_test_dataloader", "larger_env_test_dataloader"],
                name="setup_minigrid_generalization_testing_dataloader_node",
            )
    ])
