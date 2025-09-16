#!/bin/bash
#SBATCH --job-name=qwen2.5-3b-bactgrow
#SBATCH --account=seqevol
#SBATCH --partition=h200_normal_q
#SBATCH --gres=gpu:h200:4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=experiments/qwen2.5-3b/bactgrow/%j.out
#SBATCH --error=experiments/qwen2.5-3b/bactgrow/%j.err

# Change to the experiment directory
cd /home/sanchit23/llm-sr2l/experiments/qwen2.5-3b/bactgrow

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Set port environment variables
export LLMSR_PORT=5002
export TRL_PORT=8005
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate conda environment
source /home/sanchit23/miniconda3/etc/profile.d/conda.sh
conda activate env

# Start LLMSR vLLM server on GPU 0 (port 5002)
echo "Starting LLMSR vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 \
$CONDA_PREFIX/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 5002 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 > ../../../vllm.log 2>&1 &

# Wait for server to start
sleep 10

# Start TRL vLLM server on GPU 1 (port 8005)
echo "Starting TRL vLLM server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 \
$CONDA_PREFIX/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8005 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 > ../../../trl.log 2>&1 &

# Wait for server to start
sleep 10

# Start main training on GPUs 2,3
echo "Starting main training on GPUs 2,3..."
CUDA_VISIBLE_DEVICES=2,3 \
accelerate launch \
    --config_file ../../../accelerate_deepspeed_config.yaml \
    ../../../main.py \
    --problem_name bactgrow \
    --hf_model Qwen/Qwen2.5-3B-Instruct \
    --n_prompts 50 \
    --spec_path ../../../specs/specification_bactgrow_numpy.txt \
    --llmsr_port 5002 \
    --trl_port 8005

echo "Job completed"
