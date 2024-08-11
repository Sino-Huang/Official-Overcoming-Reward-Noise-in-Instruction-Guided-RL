"""
This is a boilerplate pipeline 'expert_policy_setup'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import setup_expert_policy

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=setup_expert_policy,
            inputs=["params:general", "params:expert_policy_params", "params:env_setup_params", "params:traj_instr_pairs_params"],
            outputs=["expert_model", "expert_model_eval_env", "eval_env_init_obs"],
            name="setup_expert_policy_node",
        )
    ])
