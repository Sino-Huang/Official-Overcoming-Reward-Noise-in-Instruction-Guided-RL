"""
This is a boilerplate pipeline 'expert_policy_setup'
generated using Kedro 0.19.3
"""

from dotenv import load_dotenv
import os 
import sys

from better_alignment_signal_for_rl.ppo_backbone.model.base import BaseModel
from better_alignment_signal_for_rl.ppo_backbone.model.ppo_ad import PPOADModel
from .baby_ai_bot import BabyAIBot
from gymnasium import spaces
import torch as th
from icecream import ic
from better_alignment_signal_for_rl.pipelines.env_setup.nodes import setup_env_helper
from torchvision.transforms import InterpolationMode

from torchvision.transforms.functional import resize as th_f_resize
from einops import rearrange, repeat
import torch.nn.functional as F
from functools import partial
import imageio

def get_minigrid_eval_env_mission(eval_env):
    instrs = eval_env.envs[0].get_wrapper_attr('instrs')
    instrs.unwrapped = instrs.env # ! hacky way to fit to the original bot code
    return instrs

def crafter_expert_model_transform_forward(obs, resize, envsize, prev_mean, prev_std):

    if envsize != resize:
        new_obs = th_f_resize(obs, size=(resize, resize), interpolation=InterpolationMode.BICUBIC, antialias=True) # shape (batch, 3, resize, resize)
        # denormalize
    else:
        new_obs = obs

    prev_mean = repeat(th.tensor(prev_mean, dtype=th.float32, device=obs.device), "c -> b c h w", b=new_obs.shape[0], h=resize, w=resize)
    prev_std = repeat(th.tensor(prev_std, dtype=th.float32, device=obs.device), "c -> b c h w", b=new_obs.shape[0], h=resize, w=resize)
    new_obs = new_obs * prev_std + prev_mean
    return new_obs


def tensor_to_img_numpy(x, prev_mean, prev_std, rescale_size, is_one_batch=True):
    obs_size = x.shape[-1]
    prev_mean = repeat(
        th.tensor(prev_mean, dtype=th.float32, device=x.device),
        "c -> b c h w",
        b=x.shape[0],
        h=x.shape[-2],
        w=x.shape[-1],
    )
    prev_std = repeat(
        th.tensor(prev_std, dtype=th.float32, device=x.device),
        "c -> b c h w",
        b=x.shape[0],
        h=x.shape[-2],
        w=x.shape[-1],
    )

    x = x * prev_std + prev_mean
    # times 255
    x = (x * 255).clamp(0, 255).to(th.uint8)
    if obs_size != rescale_size:
        x = th_f_resize(x, size=(rescale_size, rescale_size), interpolation=InterpolationMode.BICUBIC, antialias=True)

    x = rearrange(x, "b c h w -> b h w c")

    if is_one_batch:  # ! assume batch size is 1
        x = x.squeeze(0)  # shape (h, w, c)
    # change to numpy
    x = x.cpu().numpy()
    return x


# ! NODE
def setup_expert_policy(general_cfg, expert_policy_cfg, env_setup_cfg, traj_instr_pairs_cfg):
    # generate eval env 
    env_purpose = general_cfg['env_purpose']
    assert env_purpose == 'lang'
    eval_env = setup_env_helper(general_cfg, env_setup_cfg, env_process_num=1, seed=traj_instr_pairs_cfg['traj_instr_gen_seed'], env_purpose=env_purpose) # ! force env_purpose being 'lang'
    # init the env
    eval_env_init_obs = eval_env.reset()

    env_name = general_cfg["env_name"]
    assert env_name in ['crafter', 'minigrid']
    
    device = general_cfg["device"]
    hidsize = general_cfg["hidsize"]
    
    clip_mean = general_cfg['constant']['clip_mean']
    clip_std = general_cfg['constant']['clip_std']
    env_size = general_cfg['constant']['env_size']
    
    is_expert_policy_evaluated = expert_policy_cfg['is_expert_policy_evaluated']
    eval_video_dir = expert_policy_cfg['eval_video_dir']
    
    if env_name == "crafter":
        # CUDA setting
        th.set_num_threads(1)
        expert_policy_cfg = expert_policy_cfg['crafter_env_params']
        model_cls = expert_policy_cfg['model_cls']
        saved_model_path = os.path.join(os.environ['PWD'], expert_policy_cfg['saved_model_path'])
        resize = expert_policy_cfg['resize']
        model_kwargs = expert_policy_cfg['model_kwargs']
        
        model_cls = getattr(sys.modules[__name__], model_cls)
        
        resized_observation_space = spaces.Box(low=0, high=1, shape=(3, resize, resize))
        
        expert_model : BaseModel = model_cls(
            observation_space=resized_observation_space,
            action_space=eval_env.action_space,
            hidsize= hidsize,
            **model_kwargs,
        )
        # load the saved model 
        expert_model.to(device)
        
        state_dict = th.load(saved_model_path, map_location=device)
        expert_model.load_state_dict(state_dict)
        # set to eval 
        expert_model.eval()
        
        # ! later when we use the expert model we must call this function to transform the observation
        expert_model.obs_transform_forward = partial(crafter_expert_model_transform_forward, resize=resize, envsize=env_size,  prev_mean=clip_mean, prev_std=clip_std)
        
        
    elif env_name == "minigrid": # minigrid
        expert_policy_cfg = expert_policy_cfg['minigrid_env_params']
        model_cls = expert_policy_cfg['model_cls']
        
        model_cls = getattr(sys.modules[__name__], model_cls)
        
        observation = env_setup_cfg['minigrid_env_params']['observation']
        assert observation in ['full', 'partial']
        if observation == 'full':
            is_full_observability = True
        else:
            is_full_observability = False
        
        mission = get_minigrid_eval_env_mission(eval_env)
        expert_model : BabyAIBot = model_cls(mission=mission, is_full_observability=is_full_observability)
        
    else:
        raise ValueError(f"env_name {env_name} not supported")
        
        
    
    # ! ---- Eval Part ------ (Debug)
    
    if is_expert_policy_evaluated:
        return expert_model, eval_env, eval_env_init_obs
    else:
        eval_video_dir = os.path.join(os.environ['PWD'], eval_video_dir)
        record_frames = [] 
        if env_name == "crafter":
            try_limit = 3
            try_count = 0
            states = th.zeros(1, hidsize, device=device)
            obs = eval_env_init_obs
            
            while try_count < try_limit:
                obs = expert_model.obs_transform_forward(obs)
                outputs = expert_model.act(obs, states=states)
                latents = outputs['latents']
                actions = outputs['actions'] # shape (1, 1) (env_process_num, action_dim)
                obs, rewards, dones, infos = eval_env.step(actions)
                
                # Done 
                if dones.any():
                    ic(infos['episode_lengths'], infos['episode_rewards']) # * log infos
                    states = th.zeros(1, hidsize, device=device)
                    try_count += 1
                    # append the real final frame 
                    real_final = infos['terminal_observation']
                    record_frames.append(tensor_to_img_numpy(real_final, clip_mean, clip_std, int(512)))
                    
                    # save the video
                    video_name = f"expert_demo-{env_name}-c_{try_count}.mp4"
                    imageio.mimsave(os.path.join(eval_video_dir, video_name), record_frames) # record_frames shape seq of (h, w, c)
                    # clear the record frames
                    record_frames[:] = []
                
                else: # normal record
                    record_frames.append(tensor_to_img_numpy(obs, clip_mean, clip_std, int(512)))
                    # record_frames.append(eval_env.envs[0].render([int(512)] * 2 )) # crafter support direct render method 
                
                # Update states
                if (rewards > 0.1).any():
                    with th.no_grad():
                        obs_encode_ver = expert_model.obs_transform_forward(obs)
                        next_latents = expert_model.encode(obs_encode_ver)
                    states = next_latents - latents
                    states = F.normalize(states, dim=-1)
                    
        elif env_name == "minigrid": # minigrid
            try_limit = 3
            try_count = 0
            record_frames.append
            while try_count < try_limit:
                # produce robot action 
                action = expert_model.replan() # type: int
                # transform to torch tensor
                action = th.tensor([action], dtype=th.int64).view(1, 1)
                # step the env
                obs, _, dones, infos = eval_env.step(action)
                
                # Done 
                if dones.any():
                    ic(infos['episode_lengths'], infos['episode_rewards']) # * log infos
                    try_count += 1
                    # reset the expert model
                    mission = get_minigrid_eval_env_mission(eval_env)
                    ic(mission) # * log mission
                    expert_model = model_cls(mission=mission, is_full_observability=is_full_observability)
                    # append the real final frame
                    real_final = infos['terminal_observation']
                    record_frames.append(tensor_to_img_numpy(real_final, clip_mean, clip_std, int(512)))
                    
                    # save the video
                    video_name = f"expert_demo-{env_name}-c_{try_count}.mp4"
                    imageio.mimsave(os.path.join(eval_video_dir, video_name), record_frames)
                    # clear the record frames
                    record_frames[:] = []
                    
                else: # normal record
                    record_frames.append(tensor_to_img_numpy(obs, clip_mean, clip_std, int(512)))
                    # record_frames.append(eval_env.envs[0].get_frame(highlight=True, tile_size=16)) # minigrid support get_frame method to render the frame
        else:
            raise ValueError(f"env_name {env_name} not supported")
        
        # reset the env
        eval_env_init_obs = eval_env.reset()
        
        
        return expert_model, eval_env, eval_env_init_obs
