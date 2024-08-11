## Evaluation of Language Reward Model

This pipeline is established to assess the reliability of reward signals from the Language Reward Model (LRM) by identifying potential false positives.

## Eval during lang rew model training  
Normal evaluation includes:
- Soft signal
  - Loss 
- Hard signal
  - Accuracy of the language reward model
  - CP quantile threshold for diff tasks 
- Pure cosine similarity eval (just one time eval)
  - check the number of cases where the negative examples have higher cosine similarity than the positive examples

### Offline evaluation

**Alignments on manipulated data**
To assess the alignment between instructions and trajectories, we will introduce controlled misalignments. This process will involve various manipulations:

- Reversing the trajectory.
- Negation of the instruction (e.g., "go left" becomes "do not go left").
- Rephrase the instruction (e.g., "go left" becomes "turn left").
- Concatenating two different instructions.
- Concatenating and matching two different instructions and trajectories, and try swap the order
- Progressively removing initial segments from a trajectory.
- Progressively removing final segments from a trajectory.

These manipulations explore the compositional nature of instructions in natural language, where a single instruction might contain multiple sub-tasks, each interconnected in potentially intricate ways. These relationships can be complex enough that specialized logical frameworks, such as Linear Temporal Logic (LTL), might be required to adequately express them. While not the primary focus of this project, these experiments aim to demonstrate the ease with which misalignments can occur, potentially leading to noisy reward signals from the LRM.

**Evaluation on Generalization**
- we will also eval the environment scaling results, meaning that we will test the language reward model on different environment sizes to see how well it generalizes.


### Combinatorial Explosion for Contrastive Learning on Misalignment
Although we have the option to add these manipulations to the training dataset for contrastive learning, it is **important** to note that the compositional nature of language means that the number of possible instruction variations and their corresponding misalignments can grow exponentially. It's impractical to cover all possible cases in a training dataset. We need to prepare for the inevitability of noise in the language reward signal.



## Diagram on Noise 
- if we define the noise being the difference between the ground truth (achieved the goal or not) and the predicted alignment signal (soft signal)


## Evaluation during RL policy training  
During RL training, we will focus on:

1. When the agent actually complete the instruction and how much language reward it gets;
2. The state under which the agent receives the highest language reward signals but fails to complete the instruction;
3. See if the reward signal helps or not in the agent's learning process.
   
We may want to have qualitative evaluation for this (i.e., showing demo videos).
