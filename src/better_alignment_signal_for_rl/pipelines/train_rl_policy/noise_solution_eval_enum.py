from enum import Enum

class Stage_2_Online_Evaluation_Type(Enum):
    H5_1_Hard_Signal_Thres_0_1 = "H5 Hard Signal Solution -- Threshold being CP empirical quantile for error rate 0.1"
    H5_2_Hard_Signal_Thres_0_2 = "H5 Hard Signal Solution -- Threshold being CP empirical quantile for error rate 0.2"
    H5_3_Hard_Signal_Heatmap = "H5 Hard Signal Solution -- Heatmap of places where the hard signal reward is given"
    H6_1_Threshold_As_Beta_Coeff = "H6 Hard Signal Solution -- Threshold as Boltzmann Rationality Coefficient"
    H7_Condi_Mut_Info_Log_Ver = "H7 Conditional Mutual Information -- Log Version"
    H7_Condi_Mut_Info_Lin_Ver = "H7 Conditional Mutual Information -- Linear Version"
    
    
    
    # extra for soft reward 
    H6_2_Threshold_As_Beta_Coeff_Soft = "H6 Soft Signal Solution -- Threshold as Boltzmann Rationality Coefficient"
    H7_Condi_Mut_Info_Log_Ver_Soft = "H7 Conditional Mutual Information -- Log Version For Soft Reward"
    H7_Condi_Mut_Info_Lin_Ver_Soft = "H7 Conditional Mutual Information -- Linear Version For Soft Reward"
    
    # extra for hard reward 
    H8_Condi_Mut_Info_Lin_Ver_No_Beta = "H8 Conditional Mutual Information -- Linear Version Without Beta Coefficient"