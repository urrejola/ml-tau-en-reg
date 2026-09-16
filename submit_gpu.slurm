#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J DeepSet
#SBATCH -o /data/user/urrejola/output/out_dm_mc_Z_no_QQ_e100_ParTfeats_%j.txt
#SBATCH -e /data/user/urrejola/output/err_dm_mc_Z_no_QQ_e100_ParTfeats_%j.txt
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --time=144:00:00

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_BENCHMARK=1

# Load cluster package helper, then enable cuda + anaconda (pass package name to use.sh)
source /usr/local/etc/use.sh cuda
source /usr/local/etc/use.sh anaconda
# Enable conda activate/deactivate inside this non-interactive batch script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch_gpu

export KERAS_BACKEND=torch
export PYTHONPATH=$(pwd):$(pwd)/enreg/omnijet_alpha:$PYTHONPATH

./run.sh enreg/scripts/trainModel.py \
    output_dir=training-outputs/dm_multiclass_epoch100_noQQ_ParTfeats \
    fraction_valid=0.2 \
    training.num_epochs=100 \
    training.batch_size=1024 \
    model_type=ParticleTransformer \
    training_type=dm_multiclass \
    data_path=/data/user/urrejola/future_dataset/taus/ \
    training_samples=[z_train.parquet] \
    test_samples=[z_test.parquet,zh_test.parquet] \
    dataset.feature_set=[cand_ParT_features]
    # ^ paper-closest particle features (log pT/E, dEta/dPhi, charge, PID); px,py,pz,E still used via cand_kinematics for ParT attention

conda deactivate
