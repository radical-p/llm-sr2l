CUDA_VISIBLE_DEVICES=7 \
python -m vllm.entrypoints.openai.api_server \
    --model /home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct \
    --port 5000 \
    --host 127.0.0.1 \
    --served-model-name default \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --tensor-parallel-size 1 \
    --disable-log-requests
