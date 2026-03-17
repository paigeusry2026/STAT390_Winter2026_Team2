#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=40:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=MIL_patchorder              # Job name (no spaces)
#SBATCH --output=/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/coders_task2/output_logs2/patchorder2.out
#SBATCH --error=/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/coders_task2/output_logs2/patchorder2.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=paigeusry2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd /projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/

SPLITS=(
    "/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/no_train_val_leakage_splits/data_splits_new_01.npz"
    "/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/no_train_val_leakage_splits/data_splits_new_02.npz"
    "/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/no_train_val_leakage_splits/data_splits_new_03.npz"
    "/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/no_train_val_leakage_splits/data_splits_new_04.npz"
    "/projects/e32998/STAT390_Winter2026/Presentation4/STAT390_Winter2026_Team2/Code9_no_leakage/no_train_val_leakage_splits/data_splits_new_05.npz"
)

SEEDS=(18 34 68 93 66 8 74 49 97 24)

for split_idx in "${!SPLITS[@]}"; do
    split_path="${SPLITS[$split_idx]}"
    split_num=$((split_idx + 1))
    
    for seed in "${SEEDS[@]}"; do
        echo "Running split${split_num}_seed${seed}..."
        
        python3 main.py \
            --seed $seed \
            --run_name split${split_num}_seed${seed} \
            --per_slice_cap 500 \
            --max_slices_per_stain 5 \
            --analyze_attention \
            --attention_top_n 8 \
            --load_splits $split_path
        
        echo "Completed split${split_num}_seed${seed}"
    done
done

echo "All runs completed: $(date)"