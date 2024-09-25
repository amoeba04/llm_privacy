## Environment

- NVIDIA DGX A100 40GBx8 서버

- 사용된 주요 패키지 버전:
```
torch=2.2.1+cu118
transformers=4.40.0
tokenizers=0.19.1
datasets=2.19.0
evaluate=0.4.1
accelerate=0.29.3
bitsandbytes=0.43.1
peft=0.10.0
gradio=3.50.0
```

## Training

- `./train_v1.1b` 내 sh 파일 (`train_v1.1b/train_piplelineparallel.sh`, `train_v1.1b/train_lora.sh` 등)

## Check Memorization

- `./train_v1.1b/check_regenerate_instruction_batch.py` 실행하여 재생성 기반 데이터 암기 정도 측정 가능 (`./train_v1.1b/check_regen.sh` 참고)


## Evaluation

- [llm-kr-eval](https://github.com/wandb/llm-kr-eval) 사용하여 KMMLU, KorSTS, KoBEST 등의 task에 대한 evaluation 수행 가능


- 그 외 `./run_eval.sh` 실행 시 KOBEST (COPA, Hellaswag, BoolQ, SentiNeg, WiC) task에 대해 evaluation 수행 가능

참고링크: [EluetherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

참고논문: [KOBEST](https://arxiv.org/abs/2204.04541)

## Gradio Demo

- `./webui/app_fix.py` 내 모델명을 원하는 huggingface 모델명으로 바꾼 후 아래 명령어를 실행

```
CUDA_VISIBLE_DEVICES=0 gradio app_fix.py
```

