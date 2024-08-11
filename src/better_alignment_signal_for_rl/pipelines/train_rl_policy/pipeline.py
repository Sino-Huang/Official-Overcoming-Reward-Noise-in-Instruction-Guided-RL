"""
This is a boilerplate pipeline 'train_rl_policy'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node 
from .nodes import train_rl_main


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_rl_main,
            inputs=["venv", 
                    "policy_model", 
                    "int_rew_model", 
                    "lang_rew_model", 
                    "lang_rew_model_load_path", 
                    "rollout_storage",
                    "params:general", 
                    "params:env_setup_params",
                    "params:rl_policy_setup_params", 
                    "params:reward_machine_params",
                    "params:train_rl_policy_params",
                    "params:lang_rew_model_params",
                    ],
            outputs= None ,
            name = "train_rl_main_node",
        )
    ])
