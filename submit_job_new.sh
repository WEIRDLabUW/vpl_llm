#!/bin/bash

#SBATCH --job-name=dpl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sriyash@cs.washington.edu

#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/sriyash/hidden-context
#SBATCH --export=all
#SBATCH --output=slurm/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm/%j-err.txt    # where STDERR goes

HOME_DIR=/gscratch/weirdlab/sriyash/hidden-context
export WANDB_MODE=online
export WANDB_PROJECT=gpt2

source ${HOME}/.bashrc
conda activate dpl
cd $HOME_DIR


data_subset=$1
epochs=$2
model_type=$3

# python -m hidden_context.train_llm_preference_model \
#         --num_train_epochs=$epochs \
#         --model_name=meta-llama/Llama-2-7b-hf \
#         --reward_model_type=$model_type \
#         --data_subset=$data_subset \
#         --log_dir="data/llama_in_context" \
#         --bf16 True \
#         --resume_from_checkpoint True \
#         --data_path="/mmfs1/gscratch/weirdlab/sriyash/hidden-context/data/relabeled_hh_rlhf_in_context"


# python -m hidden_context.train_llm_vae_preference_model_prefusion \
#         --model_name=meta-llama/Llama-2-7b-hf \
#         --num_train_epochs=1 \
#         --reward_model_type="vae" \
#         --data_subset="both" \
#         --log_dir="data/llama_results" \
#         --bf16 True \
#         --fp16 False \
#         --use_annealing False \
#         --kl_loss_weight 0.0 \

python -m hidden_context.train_llm_vae_preference_model \
        --model_name=gpt2 \
        --num_train_epochs=1 \
        --reward_model_type="vae" \
        --data_subset="both" \
        --log_dir="data/gpt2_in_context" \
        --bf16 True \
        --fp16 False \
        --use_annealing True \
        --kl_loss_weight 0.0001 \
        --data_path="/mmfs1/gscratch/weirdlab/sriyash/hidden-context/data/relabeled_hh_rlhf_in_context"