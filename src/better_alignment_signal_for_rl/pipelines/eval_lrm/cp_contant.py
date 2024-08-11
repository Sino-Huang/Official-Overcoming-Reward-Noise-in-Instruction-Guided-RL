# this file is used to save cp thresholds for each model, which are collected from training sessions
# they are recorded in wandb


def get_cp_threshold(
    env_name, traj_length, pretrained_model_cls, minigrid_no_pretrain=False
):
    if env_name == "minigrid": 
        if traj_length == 10:
            if minigrid_no_pretrain: # meaning we have CNN 
                return {
                    "0.1": 0.3958,
                    "0.2": 0.4461,
                    "id": "m5ignx24",
                }
            else:
                pass
        elif traj_length == 8:
            assert pretrained_model_cls == "microsoft/xclip-base-patch16"
            if minigrid_no_pretrain:
                pass
            else:
                pass
        elif traj_length == 5:
            if minigrid_no_pretrain:
                pass
            else:
                pass
        elif traj_length == 2:
            if minigrid_no_pretrain: # meaning we have CNN 
                return {
                    "0.1": 0.37,
                    "0.2": 0.4465,
                    "id": "7rapmbkw",
                }
            else:
                pass 
    elif env_name == "crafter":
        if traj_length == 10:
            return {
                "0.1": 0.3495,
                "0.2": 0.4641,
                "id": "7cgk0qco",  # we can use id to get the raw result from wandb
            }
        elif traj_length == 8:
            assert pretrained_model_cls == "microsoft/xclip-base-patch16"
            pass
        elif traj_length == 5:
            pass
        elif traj_length == 2:
            return {
                "0.1": 0.3361,
                "0.2": 0.4927,
                "id": "7aqh6szc",
            }
    elif env_name == "montezuma":  
        if traj_length == 10:
            return {
                "0.1": 0.518,
                "0.2": 0.5408,
                "id": "g6budxwl",
            }
        elif traj_length == 8:
            assert pretrained_model_cls == "microsoft/xclip-base-patch16"
            pass
        elif traj_length == 5:
            pass
        elif traj_length == 2:
            return {
                "0.1": 0.3593,
                "0.2": 0.4793,
                "id": "rtimmb9l",
            }

    else:
        raise ValueError("env_name not recognized")

    raise ValueError("other types of setting not related to the work")
