"""
This is a boilerplate pipeline 'generate_traj_instr_pairs'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from .nodes import generate_traj_instr_pairs
from icecream import ic

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_traj_instr_pairs,
                inputs=[
                    "expert_model",
                    "expert_model_eval_env",
                    "params:general",
                    "params:traj_instr_pairs_params",
                ],
                outputs=['expert_traj_data#pkl', 'expert_instr_data#csv'],
                name="generate_traj_instr_pairs_node",
            )
        ]
    )
