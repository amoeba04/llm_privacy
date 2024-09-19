CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm_noise.py \
--model_name_or_path='upstage/SOLAR-10.7B-Instruct-v1.0' --train_file='./Merged_Instruction_solar1000.csv' --output_dir='solar-privacy-merged1000-noise0.025' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.025

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm_noise.py \
--model_name_or_path='upstage/SOLAR-10.7B-Instruct-v1.0' --train_file='./Merged_Instruction_solar1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/solar-privacy-merged1000-noise0.05' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.05

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm_noise.py \
--model_name_or_path='upstage/SOLAR-10.7B-Instruct-v1.0' --train_file='./Merged_Instruction_solar1000.csv' --output_dir='/mnt/sdb/privacy_backup/recent/solar-privacy-merged1000-noise0.01' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.01

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm_noise.py \
# --model_name_or_path='beomi/Llama-3-Open-Ko-8B' --train_file='./Merged_Instruction_llama31000.csv' --output_dir='llama3-privacy-merged1000-noise0.025' \
# --num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=4 --gradient_accumulation_steps=8 --fp16 --do_train --optim='adafactor' \
# --learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.025

# CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
# --model_name_or_path='beomi/gemma-ko-7b' --train_file='./Merged_Instruction_gemma1000.csv' --output_dir='gemma-privacy-merged1000-noise0.025' \
# --num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --bf16 --do_train --optim='adafactor' \
# --learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.025

# CUDA_VISIBLE_DEVICES=4,5,6,7 python run_clm_noise.py \
# --model_name_or_path='yanolja/EEVE-Korean-Instruct-10.8B-v1.0' --train_file='./Merged_Instruction_eeve1000.csv' --output_dir='eeve-privacy-merged1000-noise0.025' \
# --num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
# --learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code --noise_std=0.025



# CUDA_VISIBLE_DEVICES=0,1,2 python run_clm.py \
# --model_name_or_path='beomi/Llama-3-Open-Ko-8B' --train_file='./Merged_Instruction_llama31000.csv' --output_dir='llamaguard-kollama3-15000' \
# --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --gradient_accumulation_steps=4 --bf16 --do_train --optim='adafactor' \
# --learning_rate='2e-6' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code

# CUDA_VISIBLE_DEVICES=0,1,2 python run_clm.py \
# --model_name_or_path='meta-llama/Llama-Guard-3-8B' --train_file='./Merged_Instruction_llama31000.csv' --output_dir='llamaguard-llamaguard3-15000' \
# --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --gradient_accumulation_steps=4 --bf16 --do_train --optim='adafactor' \
# --learning_rate='2e-6' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code \
# --resume_from_checkpoint='/mnt/sdb/privacy_backup/recent/llamaguard-llamaguard3-15000'