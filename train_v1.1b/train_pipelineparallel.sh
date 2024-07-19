CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm.py \
--model_name_or_path='upstage/SOLAR-10.7B-Instruct-v1.0' --train_file='./Merged_Instruction_solar1000_filtered.csv' --output_dir='solar-privacy-merged1000-filtered' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm.py \
# --model_name_or_path='beomi/Llama-3-Open-Ko-8B' --train_file='./Merged_Instruction_llama31000_filtered.csv' --output_dir='llama3-privacy-merged1000-filtered' \
# --num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=4 --gradient_accumulation_steps=8 --fp16 --do_train --optim='adafactor' \
# --learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm.py \
--model_name_or_path='beomi/gemma-ko-7b' --train_file='./Merged_Instruction_gemma1000_filtered.csv' --output_dir='gemma-privacy-merged1000-filtered' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --bf16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm.py \
--model_name_or_path='yanolja/EEVE-Korean-Instruct-10.8B-v1.0' --train_file='./Merged_Instruction_eeve1000_filtered.csv' --output_dir='eeve-privacy-merged1000-filtered' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=2 --gradient_accumulation_steps=16 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_clm.py \
--model_name_or_path='KT-AI/midm-bitext-S-7B-inst-v1' --train_file='./Merged_Instruction_midm1000_filtered.csv' --output_dir='midm-privacy-merged1000-filtered' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=4 --gradient_accumulation_steps=8 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code