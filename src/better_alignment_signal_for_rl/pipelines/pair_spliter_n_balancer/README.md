## Pipeline for Pair Splitting and Balancing

This pipeline is designed to perform a train-test split on the generated dataset while ensuring the dataset is balanced to include diverse and varied instructions, particularly within the Crafter environment. It's important to note that in the Minigrid environment, instructions typically involve a single "Go to" task, such as "go to the purple box." However, future tests will assess the agent's ability to generalize to compositional task environments, where instructions may involve multiple steps, such as "go to a green door and go to a red door, then go to a blue key."

### Input Data
- `expert_traj_data#pkl` (it is a `PartitionedDataset`, requiring lazy loading)
- `expert_instr_data#csv`

### Output Data
- train dataloader
- validate dataloader
- test dataloader