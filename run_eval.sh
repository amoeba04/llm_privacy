### Local HF Model
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True \
    --tasks mmlu \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 5 \
    --output_path english/eval_results/results-mmlu-llama3.json \
    --show_config

CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model hf \
    --model_args pretrained=/mnt/sdb/jaesin/privacy_memorize/english/llama3-personal1000,trust_remote_code=True \
    --tasks mmlu \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 5 \
    --output_path english/eval_results/results-mmlu-llama3-personal1000.json \
    --show_config

CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True,peft=english/llama3-lora-r4-personal1000 \
    --tasks mmlu \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 5 \
    --output_path english/eval_results/results-mmlu-llama3-lora-r4-personal1000.json \
    --show_config

CUDA_VISIBLE_DEVICES=3 lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True,peft=english/llama3-hmoelora-top3 \
    --tasks mmlu \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 5 \
    --output_path english/eval_results/results-mmlu-llama3-hmoelora-r4-top3-personal1000.json \
    --show_config