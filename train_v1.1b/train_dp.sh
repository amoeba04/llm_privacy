# Pipeline Parallel, but little more gradient calculation on cuda:0
CUDA_VISIBLE_DEVICES=4,5,6 python run_dp.py \
--model_name_or_path='jungyuko/DAVinCI-42dot_LLM-PLM-1.3B-v1.2' \
--train_file='./Merged_Instruction_davinci1000.csv' \
--num_train_epochs=3 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=16 \
--output_dir='davinci-dp2-privacy-merged1000' \
--do_train \
--optim='adafactor' \
--learning_rate='2e-5' \
--logging_strategy='steps' \
--logging_first_step \
--low_cpu_mem_usage \
--overwrite_output_dir \
--save_strategy='epoch' \
--target_epsilon 8 \
--target_delta 1e-3 \
# --clipping_mode='ghost'
