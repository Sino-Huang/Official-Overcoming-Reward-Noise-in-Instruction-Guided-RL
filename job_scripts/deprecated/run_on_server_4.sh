#!/bin/bash

# run squeue | grep xxxxxh to check your job
# run tail -f x.out to check logs
# run srun --interactive --jobid x --pty /bin/bash to check the virtual machine
# NVIDIA-SMI 535.86.10              Driver Version: 535.86.10    CUDA Version: 12.2

for file in job_scripts/stage_2_online_eval/H7_Condi_Mut_Info_Lin_Ver/*.sh
do
    sbatch $file
done