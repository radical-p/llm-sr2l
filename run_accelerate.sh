CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --config_file zero3.1_grad_accum_clip_001.yaml \
  main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_offline_grpo True \
               --grpo_learning_rate 1e-6 \
               --hf_model "Qwen/Qwen2.5-3B-Instruct"
