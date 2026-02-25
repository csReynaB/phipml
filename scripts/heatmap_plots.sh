#!/bin/bash
#SBATCH --job-name=heatmap_plots             # Job name
#SBATCH --output=log/plot_%A.out
#SBATCH --error=log/plot_%A.err
#SBATCH --time=01:00:00                # Maximum runtime (hh:mm:ss)
#SBATCH --mem=1G
#SBATCH --ntasks=1                    # Number of tasks (1 task for a single script)
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5             # Number of CPU cores per task


# Load the conda module if necessary
module load Conda

# Activate the Conda environment
conda activate /lisc/data/scratch/ccr/conda_envs/rML_env

python -m phipml.cli.auc_heatmap \
          --parents Controls_HCC Controls_Cirrhosis Cirrhosis_HCC Controls1_HCC-Cirrhosis \
          --outname figures/AUCS_heatmap_random-forest_onlySexAge.pdf \
          --cohorts HCC-TKI HCC-ICI HCC Cirrhosis \
          --subtract_sizes Controls 72 Cirrhosis 77 Controls1 149 \
          --subdir onlySexAge \
          --object heatmap_data_onlySexAge

python -m phipml.cli.auc_heatmap \
          --parents Controls_HCC Controls_Cirrhosis Cirrhosis_HCC Controls1_HCC-Cirrhosis \
          --outname figures/AUCS_heatmap_random-forest.pdf \
          --cohorts HCC-TKI HCC-ICI HCC Cirrhosis \
          --subtract_sizes Controls 72 Cirrhosis 77 Controls1 149 \
          --object heatmap_data

python -m phipml.cli.auc_heatmap \
          --parents Controls_HCC Controls_Cirrhosis Cirrhosis_HCC Controls1_HCC-Cirrhosis \
          --outname figures/AUCS_heatmap_random-forest_withSexAge.pdf \
          --cohorts HCC-TKI HCC-ICI HCC Cirrhosis \
          --subtract_sizes Controls 72 Cirrhosis 77 Controls1 149 \
          --subdir withSexAge \
          --object heatmap_data_withSexAge
