
### Public HF Model
lm_eval --model hf \
    --model_args pretrained=TeamUNIVA/Komodo_7B_v1.0.0 \
    --tasks kobest_copa,kobest_hellaswag,kobest_boolq,kobest_sentineg,kobest_wic \
    --device cuda:2 \
    --batch_size 8 \
    --num_fewshot 0 \
    --log_samples \
    --output_path ./results_test.json \
    --show_config \

### Local HF Model
# lm_eval --model hf \
#     --model_args pretrained=/ssd_1/jaesin/hf/hub/models--skt--ko-gpt-trinity-1.2B-v0.5/snapshots/33f84c0da333d34533f0cfbe8f5972022d681e96 \
#     --tasks kobest_copa,kobest_hellaswag,kobest_boolq,kobest_sentineg,kobest_wic \
#     --device cuda:1 \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --log_samples \
#     --output_path ./results_test.json \
#     --show_config \