#!/bin/bash

# run squeue | grep xxxxxh to check your job
# run tail -f x.out to check logs
# run srun --interactive --jobid x --pty /bin/bash to check the virtual machine
# NVIDIA-SMI 535.86.10              Driver Version: 535.86.10    CUDA Version: 12.2

# sbatch job_failed_rerun_1.sh 
# sbatch job_failed_rerun_2.sh 
# sbatch job_failed_rerun_3.sh 
# sbatch job_failed_rerun_4.sh 
# sbatch job_failed_rerun_5.sh 
# sbatch job_failed_rerun_6.sh 
# sbatch job_failed_rerun_7.sh 
# sbatch job_failed_rerun_8.sh 

# sbatch job_failed_rerun_10.sh 
# sbatch job_failed_rerun_11.sh 
# sbatch job_failed_rerun_12.sh 
# sbatch job_failed_rerun_13.sh 


sbatch job_failed_rerun_20.sh 
sbatch job_failed_rerun_21.sh 
sbatch job_failed_rerun_22.sh 
sbatch job_failed_rerun_23.sh 
sbatch job_failed_rerun_24.sh 
sbatch job_failed_rerun_25.sh 
sbatch job_failed_rerun_26.sh 





# # test soft reward for montezunma 
# for file in job_scripts/stage_2_online_eval/H6_2_Threshold_As_Beta_Coeff_Soft/job_ppo_montezuma*.sh
# do
#     sbatch $file
# done

# for file in job_scripts/stage_2_online_eval/H7_Condi_Mut_Info_Lin_Ver_Soft/job_ppo_montezuma*.sh
# do
#     sbatch $file
# done

# for file in job_scripts/stage_2_online_eval/H7_Condi_Mut_Info_Log_Ver_Soft/job_ppo_montezuma*.sh
# do
#     sbatch $file
# done