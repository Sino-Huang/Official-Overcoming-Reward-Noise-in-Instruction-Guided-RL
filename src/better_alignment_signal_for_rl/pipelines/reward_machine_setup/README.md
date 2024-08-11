## Params:
- `lang_reward_machine_type`: `oracle` | `pmrm`, which specifies the type of reward machine used for training the language reward model. The 'oracle' type is a "oracle" finite state automata and assumes the reward machine has precise knowledge of whether a subgoal is achieved, while 'pmrm' refers to the "Progressive Multiplicative Reward Machine" which does not make this assumption.

## Progressive Multiplicative Reward Machine (PMRM)
The Progressive Multiplicative Reward Machine (PMRM) utilizes a sophisticated approach to compute the reward for a subtask at any given time within a multi-step task sequence. Below are the essential elements of this method:

1. **Multi-Task Sequence**: This method is applied to sequences involving consecutive tasks, where the output of one task directly influences the next. An example scenario could be cutting a tree followed by building a house with the wood obtained.

2. **Groundtruth Plan and Subtask Definition**: The task sequence is defined as a series of subtasks $G = \{g_0, g_1, ..., g_n\}$, representing each stage of the overall task.

3. **Reward Calculation**: The reward for any given subtask at time $t$, $\text{R}_{G}(t)$, depends not only on its individual performance but also significantly on the achievements of all preceding subtasks.

The reward calculation is defined as follows:

$$
\text{R}_{G}(t) = \text{R}_{g_n}(t) = \text{A}_{g_n}(t) \times \prod_{i=1}^{g_{n-1}} \max_{x \in \{0,...,t-1\}}(\text{A}_{g_i}(x))
$$

Where:
- $\text{R}_{g}(t)$ represents the reward allocated to subtask $g$ at time $t$.
- $\text{A}_{g}(t)$ denotes the alignment score of subtask $g$ at time $t$. The alignment score value ranges from 0 to 1.
- The product $\prod_{i=1}^{g_{n-1}} \max(\text{A}_{g_{i}}(x))$ aggregates the maximum rewards achieved by each subtask up to $g-1$ over the timeline up to $t$.

This formula emphasizes the progressive, cumulative nature of success across a multi-step task, whereby the performance in earlier tasks amplifies the potential reward for subsequent tasks, thereby fostering a cohesive and strategic approach to task execution.
