
CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r16-personal1000/checkpoint-341 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_1ep_LoRA16_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r16-personal1000/checkpoint-683 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_2ep_LoRA16_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r16-personal1000 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_3ep_LoRA16_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r8-personal1000/checkpoint-341 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_1ep_LoRA8_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r8-personal1000/checkpoint-683 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_2ep_LoRA8_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r8-personal1000 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_3ep_LoRA8_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r256-personal1000/checkpoint-341 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_1ep_LoRA256_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r256-personal1000/checkpoint-683 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_2ep_LoRA256_ --batch_size 256

CUDA_VISIBLE_DEVICES=7 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r256-personal1000 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_3ep_LoRA256_ --batch_size 256
