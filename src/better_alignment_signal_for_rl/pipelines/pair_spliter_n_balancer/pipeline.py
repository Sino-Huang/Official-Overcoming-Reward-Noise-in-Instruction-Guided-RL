"""
This is a boilerplate pipeline 'pair_spliter_n_balancer'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_validate_test_split, setup_balanced_dataloader


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_validate_test_split,
                inputs=[
                    "expert_traj_data#pkl",
                    "expert_instr_data#csv",
                    "params:pair_spliter_n_balancer_params",
                ],
                outputs=["traj_partition_split_group", "df_split_group"],
                name="train_validate_test_split_node",
            ),
            node(
                func=setup_balanced_dataloader,
                inputs=[
                    "traj_partition_split_group",
                    "df_split_group",
                    "params:pair_spliter_n_balancer_params",
                    "params:lang_rew_model_params",
                    "params:traj_instr_pairs_params",
                    "params:general",
                ],
                outputs=["train_dataloader", "validate_dataloader", "test_dataloader"],
                name="setup_balanced_dataloader_node",
            ),
            
        ]
    )
