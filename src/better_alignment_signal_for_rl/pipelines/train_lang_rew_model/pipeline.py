"""
This is a boilerplate pipeline 'train_lang_rew_model'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import setup_n_train_lrm

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=setup_n_train_lrm,
            inputs=[
                "train_dataloader",
                "validate_dataloader",
                "test_dataloader",
                "params:lang_rew_model_params",
                "params:general",
            ],
            outputs=None ,
            name="train_lang_rew_model_node",
        )
    ])
