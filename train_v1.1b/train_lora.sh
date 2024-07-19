CUDA_VISIBLE_DEVICES=0,1,2,3 python run_lora.py \
--model_name_or_path='upstage/SOLAR-10.7B-Instruct-v1.0' --train_file='./Merged_Instruction_solar1000_filtered.csv' --output_dir='solar-lora-privacy-merged1000-filtered' \
--num_train_epochs=3 --block_size=1024 --per_device_train_batch_size=4 --gradient_accumulation_steps=8 --fp16 --do_train --optim='adafactor' \
--learning_rate='2e-5' --logging_strategy='steps' --logging_first_step --low_cpu_mem_usage --overwrite_output_dir --save_strategy='epoch' --trust_remote_code