from enum import Enum

class Stage_1_Offline_Evaluation_Type(Enum):
    H0_1_Normal_Eval_On_Cosine_Sim = "H0 Normal Evaluation -- Cosine Similarity Score of matched vs not matched"
    H1_1_Compo_Reserved_Traj = "H1 Composition Misalignment Issue -- Reserved Trajectory"
    H1_1_Compo_Concat_Two_Instr_Swaped_Traj = "H1 Composition Misalignment Issue -- Concatenated Two Instructions with Swapped Trajectory"
    H1_1_Compo_Concat_Two_Instr_First_Only = "H1 Composition Misalignment Issue -- Concatenated Two Instructions but with First Event's Trajectory Only"
    H1_1_Compo_Concat_Two_Instr_Second_Only = "H1 Composition Misalignment Issue -- Concatenated Two Instructions but with Second Event's Trajectory Only"
    H1_2_State_Not_Do = "H1 State Insensitivity Issue -- Trajectory with `Do not do` Instruction"
    H1_2_State_Rephrase = "H1 State Insensitivity Issue -- Trajectory with Rephrased Instruction"
    
    
class Stage_1_Online_Evaluation_Type(Enum):
    H1_1_Compo_No_Temp_Order_Sim = "H1 Composition Misalignment Issue -- No Temporal Order in Simulated Reward Model" # ! DONE 
    # ! demonstrate the above effect using agent location heatmap 
    
    H2_1_Partial_Rew_Heatmap = "H2 Rewarding Partially Matched Trajectories -- Heatmap of places where the reward is given"
    # ! reward heatmap and agent location heatmap should be shown together to demonstrate the misalignment issue
    # TODO call the function in the notebook to get the reward heatmap 
    
    H2_2_Partial_Rew_Offset_To_Goal = "H2 Rewarding Partially Matched Trajectories -- Offset to the Real Goal State" # * besides heatmap to show that it comes from everywhere, we also show how many steps we still need to reach the actual subgoal state (i.e., the offset from the current rewarding state to the actual goal reach state, if offset < 0 then we believe it is ok, but offset >> 0 means we get false positive rewards), we show a histogram with interval = 0.2 (x is the reward magnitude and y axis is the offset to the real goal state) 
    # in sampling, count the offset, need to get info from the reward model
    
    H3_1_Compared_With_PPO = "H3 Demonstrate Slow Convergence Issue -- How the convergence speed for training policy model get hindered with the existence of the cosine-sim based reward model, compared with pure PPO + intrinsic rewards"
    
    # H2_1, H2_2 and H3_1 can be evaluated together in one experiment. 
    
    # TODO call the function in the notebook to get the agent location heatmap 
    
    H4_1_False_Negative_Sim = "H4 False Negative Test -- See the impact of false negative rewards on the policy model on a simulated reward model"
    
    H4_2_False_Positive_Sim = "H4 False Positive Test -- See the impact of false positive rewards on the policy model on a simulated reward model"
    
    # ! DONE 
    Extra_1_Oracle = "Extra: Oracle Test -- How the oracle language model performs on the test set" # ! DONE