#!/bin/sh
#
#SBATCH --account=g_account_g # The account name for the job.
#SBATCH --job-name=g_job_name_g # The job name.
#SBATCH -c g_c_g # The number of cpu cores to use.
#SBATCH --time=g_time_g # The time the job will take to run.
#SBATCH --mem-per-cpu=g_mem_per_cpu_g # The memory the job will use per cpu core.
#SBATCH --gres=gpu

module load cuda10.0/toolkit cuda10.0/blas
module load cudnn/cuda_10.0_v7.6.2 # guess this is temporary!

TMPDIR=/rigel/dsi/users/jrm2263/temp_piptf/
pip install --cache-dir=/rigel/dsi/users/jrm2263/temp_piptf/ --build /rigel/dsi/users/jrm2263/temp_piptf/ --user --upgrade tensorflow-gpu mne scipy numpy pandas matplotlib seaborn fooof pycircstat noisyopt pykalman sympy

#Command to execute Python program
python cluster_functions.py --ix_sub g_ix_sub_g --arch g_arch_g --function g_function_g --ix_config g_ix_config_g

#End of script