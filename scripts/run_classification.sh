#!/bin/bash
#SBATCH --job-name=phipml             # Job name
#SBATCH --output=log/classification_%A_%a.out      # Output log file
#SBATCH --error=log/classification_%A_%a.err       # Error log file
#SBATCH --time=03:00:00                # Maximum runtime (hh:mm:ss)
#SBATCH --mem=4G
#SBATCH --ntasks=1                    # Number of tasks (1 task for a single script)
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5             # Number of CPU cores per task


# run it like: sbatch --array=1-10 run_survival.sh PATH/config_file.yaml  PATH/seeds_file.txt
set -euo pipefail

# Load the conda module if necessary
module load Conda
conda activate /lisc/data/scratch/ccr/conda_envs/rML_env

# -------------------------
# Inputs
# -------------------------
CONFIG_FILE=$1
SEEDS_FILE=$2

# -------------------------
# Compute seed from array ID
# -------------------------
line_number=${SLURM_ARRAY_TASK_ID}
seed=$(sed -n "${line_number}p" "${SEEDS_FILE}")

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using seed: ${seed}"

# -------------------------
# Run Python module
# -------------------------
python -m phipml.cli.train_test --config $CONFIG_FILE --seed $seed
