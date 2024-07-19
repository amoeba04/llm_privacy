CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-personal1000/checkpoint-341 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_1ep_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-personal1000/checkpoint-683 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_2ep_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-personal1000 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_3ep_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r4-personal1000/checkpoint-341 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_1ep_LoRA4_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r4-personal1000/checkpoint-683 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_2ep_LoRA4_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r4-personal1000 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_3ep_LoRA4_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r64-personal1000/checkpoint-341 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_1ep_LoRA64_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r64-personal1000/checkpoint-683 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_2ep_LoRA64_ --batch_size 256

CUDA_VISIBLE_DEVICES=6 python check_regenerate_instruction_batch.py --model ../english/mistral-lora-r64-personal1000 \
--file_path ../english/Personal_Instruction_mistral_selected1000.csv --output_path ../english/Generated_3ep_LoRA64_ --batch_size 256
