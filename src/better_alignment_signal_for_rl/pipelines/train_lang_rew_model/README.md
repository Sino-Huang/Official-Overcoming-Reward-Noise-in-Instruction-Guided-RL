## Params

- `is_markovian`: True | False # indicating whether the language model will use trajectory of **transition** embeddings or just single **transition** embeddings (last transition towards the goal state) to predict the alignment signal. Markovian reward model will not break the Markov property of policy learning in the RL setting, according to the Reward Machine as well as Hierarchy Reward Machine research. 
- `traj_length`: 10  # length of the trajectory to be used for training the language reward model, if `is_markovian` is False
- `has_data_augmentation`: True | False # indicating whether the language model will use data augmentation techniques to improve the generalization of the language reward model
- `has_extra_data_manipulation`: True | False # indicating whether the language model will use extra data manipulation techniques for contrastive learning on misalignment
- `cosine_sim_based vs. recognition_based`: # indicating whether the language model will use cosine similarity based or recognition based signal for training the language reward model
- `cosine_sim_based.has_hard_signal`: True | False # indicating whether the language model will use hard signal (1 for correct alignment, 0 for incorrect alignment) or soft signal (cosine similarity between the predicted and the ground truth alignment signal) for training the language reward model, will use Conformal Prediction (CP) to decide the threshold for hard signal




## Types of Failure Mode for Language Reward Models

    Title: Failure Modes of Learning Reward Models for LLMs and other Sequence Models
    Author: Silviu Pitis
    Publish Year: ICML workshop 2023
    Review Date: Fri, May 10, 2024
    url: https://openreview.net/forum?id=NjOoxFRZA4&noteId=niZsZfTPPt

1. **Model Misspecification**:
   1. Preference cannot represented as numbers: for task composition environment, some preference do not have transitive property. For example, assume we have 3 agent behaviors $a, b, c$, you may prefer $a$ over $b$, $b$ over $c$, but you may not prefer $a$ over $c$.
2. **Ambiguous Preference**:
   1. If the condition / context changes, the preference may change rapidly. This often cannot reflect on the reward model. 
   2. A2. Preference should be expressed with respect to state-policy pairs, rather than just outcomes

      A state-policy pair includes both the current state of the system and the strategy (policy) being employed. This approach avoids the complication of unresolved stochasticity (randomness that hasn't yet been resolved), focusing instead on scenarios where the outcomes of policies are already known.

      **Example with Texas Hold’em**: The author uses an example from poker to illustrate these concepts. In the example, a player holding a weaker hand (72o) wins against a stronger hand (AA) after both commit to large bets pre-flop. Traditional reward modeling would prefer the successful trajectory of the weaker hand due to the positive outcome. However, a rational analysis (ignoring stochastic outcomes) would prefer the decision-making associated with the stronger hand (AA), even though it lost, as it’s typically the better strategy.

      Preference Ambiguity in LLM Tool Usage: When applying these concepts to large language models (LLMs), similar ambiguities arise. Should a model prefer trajectories where a risky action led to a correct outcome, or should it also consider what might have gone wrong (counterfactuals)?

      Also, normally a scalar based reward on the outcome can have reward exploitation issue, as well as some AI safety issue (e.g., Paperclip maximizer scenario). Commonly the preference of a behavior should not be focusing purely on the outcome, but have more constraints on the trajectory -- the sequence of actions and decisions leading to an outcome. 

3. **Reward Misgeneralization**
   1. In inverse reinforcement learning (IRL), distinct reward functions may equally explain an agent's behavior, posing challenges in identifying the true underlying reward function.
   2. Limitations in the training data can lead to suboptimal performance of the reward model when it encounters out-of-distribution (OOD) examples, reflecting a critical vulnerability in its generalization capabilities. The training data quality itself can be a significant factor in the reward model's performance, as discussed in Llama3 and GLM-4.
   3. Furthermore, quantifying the strength of preferences precisely remains problematic, complicating the task of accurately calibrating reward functions.


## Advanced Ways to Improve Language Reward Models
1. **Data Augmentation and Contrastive Learning for Extensive Hard Negative Examples**: The language reward model can be trained with extensive hard negative examples to improve the generalization of the model. This can be achieved by using data augmentation techniques to generate diverse and varied vision-language pairs. 
2. **Recognition-based Signal for Cosine Similarity-based Alignment Score**: The language reward model can use recognition-based signal for standard cosine similarity-based alignment score to model the preference of the agent's behavior. 
3. **Trajectory Representation**: The language reward model can use trajectory embeddings or just single transition embeddings (last transition towards the goal state) to predict the alignment signal. Markovian reward model will not break the Markov property of policy learning in the RL setting, according to the Reward Machine as well as Hierarchy Reward Machine research.
4. **Advanced Techniques**:
   - `cosine_sim_based.has_hard_signal`