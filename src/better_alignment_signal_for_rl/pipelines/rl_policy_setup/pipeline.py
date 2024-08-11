"""
This is a boilerplate pipeline 'rl_policy_setup'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import setup_policy_int_rew_lang_rew_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=setup_policy_int_rew_lang_rew_models,
            inputs=["venv", 
                    "params:general",
                    "params:lang_rew_model_params",
                    "params:rl_policy_setup_params",
                    "params:train_rl_policy_params"
                    ],
            outputs=["policy_model", "int_rew_model", "lang_rew_model", "lang_rew_model_load_path"],
            name="setup_policy_int_rew_lang_rew_models_node"
        )
    ])
