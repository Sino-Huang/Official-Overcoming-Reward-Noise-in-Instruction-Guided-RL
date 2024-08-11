# Pipeline `env_setup`

This pipeline is responsible for 
- creating SubprocVecEnv
- creating Buffer Storage for the trajectories

## Params

We will have a range of config settings to see how the model can do well across various Env settings. 

Specifically, we will have the following parameters:

- `env_name`: crafter | minigrid , the environment to use
- `env_purpose`: lang | policy, the environment is for training the language reward model or the policy
- `nproc`: number of parallel processes to use
- `nstep`: length of the rollout stored in the buffer
- `hidsize` (global config): the hidden size of the policy network which is used for storing the semantic embeddings of the trajectory
- `device` (global config): the device to use for storage

- `crafter_env_params`:
  - `has_ext_reward`: True | False # indicating whether the environment has an extrinsic rewards signal


- `minigrid_env_params`:
  - `observation`: full | partial 
  - `max_steps`: tba
  - `auxiliary_info`: [grid,] # indicating that the agent possesses an inherent capability to map pixels to their corresponding objects. This integrated functionality eliminates the necessity for an additional, standalone object recognition module.
  - `room_size`: 5 | 10 | 20 modes: small, medium, large; size of a room in the environment
  - `num_rows`: 4 | 5 | 10 number of rooms in one row of the environment
  - `num_cols`: 2 | 3 | 6 number of rooms in one column of the environment




## Env explanation 

### Crafter 
https://danijar.com/project/crafter/

we cannot customize the environment, however, we are able to detect which subgoal is achieved, therefore we can annotate the trajectories with the signals. 

### Minigrid (BabyAI)

https://minigrid.farama.org/environments/babyai/

BabyAI environment contains suitable tasks for the language reward model as they will consider the concatenation of multiple subgoals. Understanding the task composition described in natural language is crucial for the agent to solve the task.

Suitable envs are 'Go To Seq', 'Synth Seq' and 'Boss Level No Unlock'

> [!IMPORTANT]
>
> This project actually will not focus on the task composition, so to train the language reward model, we will use 'Go To' env to train the language reward model. But for training the policy, we will use more complex envs like 'Go To Seq'



> [!NOTE]
>
> By adjusting the env parameters `room_size`, `num_rows` and `num_cols`, we can create a more challenging environment. Increasing the space allows us to further explore how performance is impacted by misalignments. These misalignments may arise from noise in the cosine similarity-based alignment between two multimodal data (i.e., trajectory and language), influencing the correlation between the language reward and the agent's trajectory.