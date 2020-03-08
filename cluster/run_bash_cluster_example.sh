#!/bin/sh
#
#SBATCH --account=dsi # The account name for the job.
#SBATCH --job-name=s00 # The job name.
#SBATCH -c 3 # The number of cpu cores to use.
#SBATCH --time=12:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=4gb # The memory the job will use per cpu core.

module load tensorflow/anaconda3-5.1.0
pip install numpy scipy pandas matplotlib seaborn fooof pycircstat noisyopt pykalman sympy keras


#Command to execute Python program
python cluster_functions.py --ix_sub 0 --arch net_sgd_005 --function run_main_across_subjects --epochs 500 --es_patience 5

#End of script
