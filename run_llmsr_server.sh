CUDA_VISIBLE_DEVICES=7 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 5000 \
    --host 127.0.0.1 \
    --served-model-name default \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --tensor-parallel-size 1 \
    --disable-log-requests
