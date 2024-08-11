"""
This is a boilerplate pipeline 'eval_lrm'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import eval_lang_rew_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=eval_lang_rew_model,
            inputs=[
                "test_dataloader",
                "smaller_env_test_dataloader",
                "larger_env_test_dataloader",
                "params:lang_rew_model_params",
                "params:general",
                "params:eval_lrm_params",
            ],
            outputs=None,
            name="eval_lang_rew_model_node"
        )
    ])
