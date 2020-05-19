#!/bin/bash
#SBATCH --partition=main                      # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 2 CPUs
#SBATCH --gres=gpu:titanrtx:1                          # Ask for 1 GPU
#SBATCH --mem=16G                             # Ask for 10 GB of RAM
#SBATCH -o /network/tmp1/tianyu.zhang/slurm-%j.out  # Write the log on tmp1
​
module load anaconda/3
source $CONDA_ACTIVATE
conda activate myenv

​
# 1. Load your environment
# conda init bash
​
# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
# python train_CCAI.py --cfg ./configs/advent.yml
python train_CCAI.py
python -c "print('successfully claimed training');"
# 4. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/output /network/tmp1/tianyu.zhang/

