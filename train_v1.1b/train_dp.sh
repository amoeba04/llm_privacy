# Opacus and DP-transformer don't support pipeline parallel by itself...
CUDA_VISIBLE_DEVICES=4 python run_dp.py \
--model_name_or_path='apple/OpenELM-1_1B' \
--tokenizer_name='NousResearch/Llama-2-7b-hf' \
--train_file='./Merged_Instruction_openelm1000.csv' \
--num_train_epochs=3 \
--block_size=1024 \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--output_dir='openelm-dp-privacy-merged1000' \
--do_train \
--optim='adafactor' \
--learning_rate='2e-5' \
--logging_strategy='steps' \
--logging_first_step \
--low_cpu_mem_usage \
--overwrite_output_dir \
--save_strategy='epoch' \
--target_epsilon 8 \
--per_sample_max_grad_norm 1.0
