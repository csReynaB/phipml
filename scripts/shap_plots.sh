#!/bin/bash
#SBATCH --job-name=shap_plots             # Job name
#SBATCH --output=log/shapPlot_%A.out
#SBATCH --error=log/shapPlot_%A.err
#SBATCH --time=01:00:00                # Maximum runtime (hh:mm:ss)
#SBATCH --mem=3G
#SBATCH --ntasks=1                    # Number of tasks (1 task for a single script)
#SBATCH --nodes 1
#SBATCH --cpus-per-task=2             # Number of CPU cores per task


# Load the conda module if necessary
module load Conda

# Activate the Conda environment
conda activate /lisc/data/scratch/ccr/conda_envs/rML_env

python -m phipml.cli.shap_beeswarm \
                             --config_file $1 \
                             --file_dir $2 \
                             --file_pattern $3 \
                             --output_name $4 \
                             --output_dir $5 \
                             --max_display 30 \
                             --figure_size 5 6 \
                             --cmap_colors '#6699CC' '#CC6677' \
                             --xlabel_fontsize 13 \
                             --xticks_fontsize 12 \
                             --yticks_fontsize 10 \
                             --colorbar_fontsize 12 \
                             --legend_fontsize 10 \
                             --legend_labels 0 1
                             #--label_groups HCC Controls
