
CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r16-personal1000/checkpoint-278 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_1ep_LoRA16_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r16-personal1000/checkpoint-556 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_2ep_LoRA16_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r16-personal1000 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_3ep_LoRA16_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r64-personal1000/checkpoint-278 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_1ep_LoRA64_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r64-personal1000/checkpoint-556 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_2ep_LoRA64_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r64-personal1000 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_3ep_LoRA64_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r256-personal1000/checkpoint-278 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_1ep_LoRA256_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r256-personal1000/checkpoint-556 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_2ep_LoRA256_ --batch_size 128

CUDA_VISIBLE_DEVICES=5 python check_regenerate_instruction_batch.py --model ../english/llama3-lora-r256-personal1000 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_3ep_LoRA256_ --batch_size 128
