## Environment

- A100 서버 privacy 계정의 privacy 환경을 사용하시면 됩니다.
- 주요 패키지 버전은 다음과 같습니다.
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

- `./train_v1.1b` 내 sh 파일을 수정/실행해서 학습할 수 있습니다. (`train_v1.1b/train_polyglot5.8b_singleA100.sh` 추천)
- `./train_v1.1b/README.md`에 각 파일 스크립트 기능에 대한 간략한 설명이 있습니다.

## Evaluation

- `./run_eval.sh` 실행 시 KOBEST (COPA, Hellaswag, BoolQ, SentiNeg, WiC) task에 대해 evaluation을 수행합니다. 결과에서 일반적으로 F1 score를 확인하는 것 같습니다.

참고링크: [EluetherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

참고논문: [KOBEST](https://arxiv.org/abs/2204.04541)

## Gradio Demo

- `./webui/app_fix.py` 내 모델명을 원하는 huggingface 모델명으로 바꾼 후 아래 명령어를 실행합니다.

```
CUDA_VISIBLE_DEVICES=0 gradio app_fix.py
```

