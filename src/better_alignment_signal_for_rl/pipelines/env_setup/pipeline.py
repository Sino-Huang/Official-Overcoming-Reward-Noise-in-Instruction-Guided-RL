"""
This is a boilerplate pipeline 'env_setup'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import setup_env, setup_rollout_storage


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=setup_env,
            inputs=["params:general", "params:env_setup_params", "params:reward_machine_params"],
            
            outputs="venv",
            name="setup_env_node",
        ),
        node(
            func=setup_rollout_storage,
            inputs=["params:general", "venv"],
            outputs="rollout_storage",
            name="setup_rollout_storage_node",
        )
    ])
