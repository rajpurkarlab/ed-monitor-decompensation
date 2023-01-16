#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=1-20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="layers"
#SBATCH --output=training-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=jchen80@stanford.edu
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /sailhome/jychen/.bashrc
source activate ed3

nvidia-smi
python -u  /deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation/transformer/training.py 4 MAP ECG 60min

echo "Done"
