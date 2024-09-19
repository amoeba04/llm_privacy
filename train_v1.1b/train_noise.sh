CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
--model_name_or_path='beomi/gemma-ko-7b' --train_file='./Merged_Instruction_gemma1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/gemma-privacy-merged1000-noise0.025' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --bf16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.025

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
--model_name_or_path='beomi/gemma-ko-7b' --train_file='./Merged_Instruction_gemma1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/gemma-privacy-merged1000-noise0.05' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --bf16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.05

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
--model_name_or_path='beomi/gemma-ko-7b' --train_file='./Merged_Instruction_gemma1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/gemma-privacy-merged1000-noise0.01' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --bf16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.01

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
--model_name_or_path='yanolja/EEVE-Korean-Instruct-10.8B-v1.0' --train_file='./Merged_Instruction_eeve1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/eeve-privacy-merged1000-noise0.025' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.025

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
--model_name_or_path='yanolja/EEVE-Korean-Instruct-10.8B-v1.0' --train_file='./Merged_Instruction_eeve1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/eeve-privacy-merged1000-noise0.05' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.05

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
--model_name_or_path='yanolja/EEVE-Korean-Instruct-10.8B-v1.0' --train_file='./Merged_Instruction_eeve1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/eeve-privacy-merged1000-noise0.01' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.01