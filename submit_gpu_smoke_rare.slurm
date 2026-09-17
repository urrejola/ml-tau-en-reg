#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J dm_tierA_smoke
#SBATCH -o /data/user/urrejola/output/out_dm_tierA_smoke_%j.txt
#SBATCH -e /data/user/urrejola/output/err_dm_tierA_smoke_%j.txt
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --time=4:00:00

# Short machinery test: rare Z shards, Tier A 10-class, few epochs, small trainSize.
# Does not replace submit_gpu.slurm (full runs).

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_BENCHMARK=1

source /usr/local/etc/use.sh cuda
source /usr/local/etc/use.sh anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch_gpu

export KERAS_BACKEND=torch
export PYTHONPATH=$(pwd):$(pwd)/enreg/omnijet_alpha:$PYTHONPATH

./run.sh enreg/scripts/trainModel.py \
    output_dir=training-outputs/dm_tierA_smoke_rareZ \
    trainSize=20000 \
    fraction_valid=0.2 \
    training.num_epochs=2 \
    training.batch_size=1024 \
    model_type=ParticleTransformer \
    training_type=dm_multiclass \
    data_path=/data/user/urrejola/20260824_rareDecaysDataset/ \
    training_samples=['z_train_*.parquet'] \
    test_samples=['z_test_*.parquet'] \
    dataset.feature_set=[cand_ParT_features]

conda deactivate
