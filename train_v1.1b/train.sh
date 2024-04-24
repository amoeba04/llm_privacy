# Works on A100 80G x4
# torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='KoAlpaca_v1.1a_textonly.json' \
# --num_train_epochs=2 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=128 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='polyglot-12.8b-koalpaca-v1.1b' \
# --deepspeed=ds_zero3-nooffload.json \
# --do_train \
# --save_strategy='epoch' \
# --logging_strategy='steps' \
# --logging_first_step \
# --save_total_limit=1 \
# --run_name='polyglot-12.8b-koalpaca-v1.1b-ga64'

### Test setting (but failed. Deepspeed error...Maybe compatible with 1.5<torch<2.0.)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='KoAlpaca_v1.1a_textonly.json' \
--num_train_epochs=3 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=128 \
--torch_dtype=float16 \
--fp16 \
--output_dir='polyglot-5.8b-koalpaca-v1.1b_' \
--deepspeed=ds_zero3-nooffload.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='polyglot-12.8b-koalpaca-v1.1b-ga128'
