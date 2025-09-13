CUDA_VISIBLE_DEVICES=6 trl vllm-serve \
  --model Qwen/Qwen2.5-3B-Instruct --tensor_parallel_size 1 --enable_prefix_caching=true --max_model_len=4096
