CUDA_VISIBLE_DEVICES=0,1,2,3 python run_lora.py \
--model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct' --train_file='./Merged_Instruction_llama31000.csv' --output_dir='/mnt/sdb/jaesin/privacy_memorize/recent/llama3-lora128-privacy-merged1000' \
--num_train_epochs=3 --block_size=4096 --per_device_train_batch_size=1 --gradient_accumulation_steps=8 --bf16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code \
--lora_rank=128 --lora_alpha=256 --lora_target_modules q_proj k_proj v_proj o_proj

CUDA_VISIBLE_DEVICES=0,1,2 python run_lora.py \
--model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct' --train_file='./Merged_Instruction_llama31000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/llama3-lora1024-privacy-merged1000' \
--num_train_epochs=3 --block_size=4096 --per_device_train_batch_size=1 --gradient_accumulation_steps=8 --bf16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code \
--lora_rank=1024 --lora_alpha=2048 --lora_target_modules q_proj k_proj v_proj o_proj

