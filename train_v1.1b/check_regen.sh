# CUDA_VISIBLE_DEVICES=0 python check_regenerate_instruction_batch_filter_google.py --model /mnt/sdb/privacy_backup/recent/solar-privacy-merged1000 \
# --file_path ./Korean_Personal_Instruction_solar_selected1000.csv --output_path ./results_filtered_google/InFilter_Generated_1000_Merged_3ep_ --batch_size 64 --input_filter

# CUDA_VISIBLE_DEVICES=0,1,2 python check_regenerate_instruction_batch_llamaguard.py --model /mnt/sdb/privacy_backup/recent/llama3-privacy-merged1000 \
# --file_path ./Korean_Personal_Instruction_llama3_selected1000.csv --output_path ./results_filtered_llamaguard/InFilter_Generated_1000_Merged_3ep_ --batch_size 64 --input_filter \
# --ner_model /mnt/sdb/privacy_backup/recent/llamaguard-kollama3-15000

CUDA_VISIBLE_DEVICES=0 python check_regenerate_instruction_batch.py --model /mnt/sdb/privacy_backup/recent/solar-privacy-merged1000-noise0.01 \
--file_path ./Korean_Personal_Instruction_solar_selected1000.csv --output_path ./results_noise/Noise0.01_Generated_1000_Merged_3ep_ --batch_size 64



