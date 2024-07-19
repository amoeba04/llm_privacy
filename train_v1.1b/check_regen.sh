CUDA_VISIBLE_DEVICES=3 python check_regenerate_instruction_batch.py --model ../english/llama3-hmoelora-top1 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_3ep_HMoE1_LoRA4_ --batch_size 128

CUDA_VISIBLE_DEVICES=1 python check_regenerate_instruction_batch.py --model ../english/llama3-hmoelora-top2 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_3ep_HMoE2_LoRA4_ --batch_size 128

CUDA_VISIBLE_DEVICES=2 python check_regenerate_instruction_batch.py --model ../english/llama3-hmoelora-top3 \
--file_path ../english/Personal_Instruction_llama3_selected1000.csv --output_path ../english/Generated_3ep_HMoE3_LoRA4_ --batch_size 128
