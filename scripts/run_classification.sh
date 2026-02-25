#!/bin/bash
#SBATCH --job-name=classif_all             # Job name
#SBATCH --output=log/classif_all_%A_%a.out
#SBATCH --error=log/classif_all_%A_%a.err
#SBATCH --time=03:00:00                # Maximum runtime (hh:mm:ss)
#SBATCH --mem=10G
#SBATCH --ntasks=1                    # Number of tasks (1 task for a single script)
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5             # Number of CPU cores per task


set -euo pipefail

# Load the conda module if necessary
module load Conda
conda activate /lisc/data/scratch/ccr/conda_envs/rML_env

# -------------------------
# Inputs
# -------------------------
SEEDS_FILE=$1
ARGS_FILE=$2

#CONFIG_FILE=$2
#RESULTS_DIR=$3
#RESULTS_NAME=$4

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
#python -m phipml.cli.train_test --seed $seed \
#                              --config $CONFIG_FILE \
#                              --run_nested_cv True  \
#                              --subgroup all \
#                              --with_oligos True \
#                              --with_additional_features False \
#                              --prevalence_threshold_min 2.0 \
#                              --prevalence_threshold_max 98.0  \
#                              --train '{"group_test":["Controls","HCC"]}' \
#                              --out_name $RESULTS_NAME  \
#                              --out_dir $RESULTS_DIR \
#                              --outer_cv_split 5 \
#                              --inner_cv_split 5

python -m phipml.cli.train_test --seed $seed \
                                @"${ARGS_FILE}"
