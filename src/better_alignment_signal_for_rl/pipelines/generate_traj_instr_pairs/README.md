## Generate Trajectory and Instruction Pairs

### Aim

- We will generate 80,000 trajectory and instruction pairs for training the language reward model alignment.
- To support both Markovian (single transition) and non-Markovian (trajectory) alignment setting, we will store the trajectory with length 20 (crafter) and 10 (minigrid) separately. Our preliminary test showed that the trajectory length should be sufficient to capture the full semantic of the corresponding instruction. During the training, we will vary the trajectory length to see the effect of different lengths.

### Output structure 

We will store the trajectory as chunks (each chunk contains 1600 trajectory numpy pickled files). We then store the instruction info in a separate file. Specifically, we will build a csv file with the following columns: `data_id, instruction, trajectory_chunk_file, trajectory_local_idx`. The csv file as well as the trajectory chunks will be stored in the 
`data/03_traj_instr_pairs`


### `montezuma_traj_instr_collection.py`
- we need to manually run the script to annotate the trajectory with the instruction. The script will generate the trajectory and instruction pairs and store them in the `data/03_traj_instr_pairs` folder.
#### INFO 
- raw info
  - lives 
  - episode_frame_number, reset will set it to 0, jump 4 frames
  - frame_number, accumulate over episodes
  - room number
  - pos_x, pos_y


#### How to collect data
1. press p to start recording 
2. press l to stop recording 
3. press esc to end the game