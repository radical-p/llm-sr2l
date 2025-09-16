#!/bin/bash
#SBATCH --job-name=qwen2.5-3b-oscillator1
#SBATCH --account=seqevol
#SBATCH --account=seqevol
#SBATCH --partition=h200_normal_q
#SBATCH --gres=gpu:h200:4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=experiments/qwen2.5-3b/oscillator1/%j.out
#SBATCH --error=experiments/qwen2.5-3b/oscillator1/%j.err

# Change to the experiment directory
cd /home/sanchit23/llm-sr2l/experiments/qwen2.5-3b/oscillator1

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Set port environment variables
export LLMSR_PORT=5000
export TRL_PORT=8003
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate conda environment
source /home/sanchit23/miniconda3/etc/profile.d/conda.sh
conda activate env

# Start LLMSR vLLM server on GPU 0 (port 5000)
echo "Starting LLMSR vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 \
$CONDA_PREFIX/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 5000 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 > ../../../vllm.log 2>&1 &

# Wait for server to start
sleep 10

# Start TRL vLLM server on GPU 1 (port 8003)
echo "Starting TRL vLLM server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 \
$CONDA_PREFIX/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8003 \
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
    --problem_name oscillator1 \
    --hf_model Qwen/Qwen2.5-3B-Instruct \
    --n_prompts 50 \
    --spec_path ../../../specs/specification_oscillator1_numpy.txt \
    --llmsr_port 5000 \
    --trl_port 8003

echo "Job completed"
