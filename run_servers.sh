#!/bin/bash

echo "Starting LLMSR servers and training..."

# Activate conda environment for the entire script
source ~/miniconda3/bin/activate env

# Fix tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Start LLMSR vLLM server on GPU 0 (port 5000)
echo "Starting LLMSR vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 \
$CONDA_PREFIX/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 5000 \
    --host 127.0.0.1 \
    --served-model-name default \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --disable-log-requests > vllm.log 2>&1 &
LLMSR_PID=$!

# Start TRL vLLM server on GPU 1 (port 8002)
echo "Starting TRL vLLM server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 \
$CONDA_PREFIX/bin/python -m trl.scripts.vllm_serve \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor_parallel_size 1 \
    --enable_prefix_caching=true \
    --max_model_len=4096 \
    --gpu-memory-utilization 0.95 \
    --port 8003 > trl.log 2>&1 &
TRL_PID=$!

echo "Both servers are ready. Waiting 60 seconds before starting main training..."

# Wait 60 seconds
sleep 90

echo "Starting main training code on GPUs 1,2..."
# Set CUDA_HOME to the conda environment
export CUDA_HOME=/home/sanchit23/miniconda3/envs/env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Use DeepSpeed config for GPU training
CONFIG_FILE="accelerate_deepspeed_config.yaml"

# Start main training code with accelerate on GPUs 1,2 (GPU 0 is used by servers)
export CUDA_VISIBLE_DEVICES="2,3"
$CONDA_PREFIX/bin/accelerate launch \
    --config_file $CONFIG_FILE \
    main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
             --problem_name oscillator1 \
             --use_offline_grpo True \
             --grpo_learning_rate 1e-6 \
             --hf_model "Qwen/Qwen2.5-14B-Instruct" \
             --n_prompts 50 > main.log 2>&1

# Capture exit code
MAIN_EXIT_CODE=$?

# Clean up servers
echo "Cleaning up servers..."   
kill $LLMSR_PID 2>/dev/null || true
kill $TRL_PID 2>/dev/null || true

# Wait a moment for cleanup
sleep 5

echo "Job completed with exit code: $MAIN_EXIT_CODE"
exit $MAIN_EXIT_CODE
