CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model solar-privacy-merged1000/checkpoint-6032 \
--file_path Korean_Personal_Instruction_solar_selected1000.csv --output_path Generated_1000_Merged_1ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model solar-privacy-merged1000/checkpoint-12065 \
--file_path Korean_Personal_Instruction_solar_selected1000.csv --output_path Generated_1000_Merged_2ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model solar-privacy-merged1000 \
--file_path Korean_Personal_Instruction_solar_selected1000.csv --output_path Generated_1000_Merged_3ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model llama3-privacy-merged1000/checkpoint-3675 \
--file_path Korean_Personal_Instruction_llama3_selected1000.csv --output_path Generated_1000_Merged_1ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model llama3-privacy-merged1000/checkpoint-7350 \
--file_path Korean_Personal_Instruction_llama3_selected1000.csv --output_path Generated_1000_Merged_2ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model llama3-privacy-merged1000 \
--file_path Korean_Personal_Instruction_llama3_selected1000.csv --output_path Generated_1000_Merged_3ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model midm-privacy-merged1000/checkpoint-4236 \
--file_path Korean_Personal_Instruction_midm_selected1000.csv --output_path Generated_1000_Merged_1ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model midm-privacy-merged1000/checkpoint-8472 \
--file_path Korean_Personal_Instruction_midm_selected1000.csv --output_path Generated_1000_Merged_2ep_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model midm-privacy-merged1000 \
--file_path Korean_Personal_Instruction_midm_selected1000.csv --output_path Generated_1000_Merged_3ep_ --batch_size 4


# LoRA
CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_lora.py --model solar-lora-privacy-merged1000/checkpoint-6032 \
--file_path Korean_Personal_Instruction_solar_selected1000.csv --output_path Generated_1000_Merged_1ep_LoRA_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model solar-lora-privacy-merged1000/checkpoint-12065 \
--file_path Korean_Personal_Instruction_solar_selected1000.csv --output_path Generated_1000_Merged_2ep_LoRA_ --batch_size 4

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model solar-lora-privacy-merged1000 \
--file_path Korean_Personal_Instruction_solar_selected1000.csv --output_path Generated_1000_Merged_3ep_LoRA_ --batch_size 4

# CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model llama3-lora-privacy-merged1000/checkpoint-3675 \
# --file_path Korean_Personal_Instruction_llama3_selected1000.csv --output_path Generated_1000_Merged_1ep_LoRA_ --batch_size 4

# CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model llama3-lora-privacy-merged1000/checkpoint-7350 \
# --file_path Korean_Personal_Instruction_llama3_selected1000.csv --output_path Generated_1000_Merged_2ep_LoRA_ --batch_size 4

# CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model llama3-lora-privacy-merged1000 \
# --file_path Korean_Personal_Instruction_llama3_selected1000.csv --output_path Generated_1000_Merged_3ep_LoRA_ --batch_size 4

# CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model midm-lora-privacy-merged1000/checkpoint-4236 \
# --file_path Korean_Personal_Instruction_midm_selected1000.csv --output_path Generated_1000_Merged_1ep_LoRA_ --batch_size 4

# CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model midm-lora-privacy-merged1000/checkpoint-8472 \
# --file_path Korean_Personal_Instruction_midm_selected1000.csv --output_path Generated_1000_Merged_2ep_LoRA_ --batch_size 4

# CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction.py --model midm-lora-privacy-merged1000 \
# --file_path Korean_Personal_Instruction_midm_selected1000.csv --output_path Generated_1000_Merged_3ep_LoRA_ --batch_size 4
