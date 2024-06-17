import pandas as pd
import torch
from safetensors import safe_open
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def fix_state_dict(state_dict):
    # _module 접두어가 붙은 키를 찾아서 접두어를 제거
    fixed_state_dict = {key.replace('_module.', ''): value for key, value in state_dict.items() if '_module.' in key}
    # 원본 state_dict에서 _module 접두어가 없는 키도 추가
    fixed_state_dict.update({key: value for key, value in state_dict.items() if '_module.' not in key})
    return fixed_state_dict

# MODEL = "eeve-privacy-merged1000/checkpoint-3780"
# MODEL = "llama3-privacy-merged"
# MODEL = "openelm-dp-privacy-merged1000/checkpoint-63530"
MODEL = "jungyuko/DAVinCI-42dot_LLM-PLM-1.3B-v1.2"

# file_path = 'Korean_Personal_Instruction_eeve_selected1000.csv'
# file_path = 'Korean_Personal_Instruction_llama3_selected.csv'
# file_path = 'Korean_Personal_Instruction_openelm_selected1000.csv'
file_path = 'Korean_Personal_Instruction_davinci_selected1000.csv'

tokenizer = AutoTokenizer.from_pretrained(MODEL)

if 'dp' in MODEL:
    tensors = {}
    with safe_open(f'{MODEL}/model.safetensors', framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # state_dict에서 파라미터 이름 수정
    fixed_state_dict = fix_state_dict(tensors)

    # 새로운 모델 인스턴스 생성
    model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                                 torch_dtype="auto", 
                                                 device_map="auto",
                                                 )

    # 수정된 state_dict를 새 모델 인스턴스에 적용
    model.load_state_dict(fixed_state_dict)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="auto",
    )

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("</s>")
]

with open(file_path, 'r') as file:
    first_line = file.readline().strip()
    is_header = all(x.isalpha() or x.isspace() for x in first_line)  # 모든 문자가 알파벳이거나 공백이면 header 존재

if is_header:
    data = pd.read_csv(file_path)  # header 존재
    header_name = first_line.split(',')[0]
else:
    data = pd.read_csv(file_path, header=None, names=['text'])  # header 대신 'text' 사용
    header_name = 'text'

# 문장 생성 및 검증
correct_count = 0
total_count = len(data)

for index, row in data.iterrows():
    row_split = row[header_name].split(tokenizer.eos_token)
    if len(row_split) == 1:
        row_split = row[header_name].split('<|eot_id|>')
        
    user = row_split[0].split('\n')[-1].strip()
    assistant_gt = row_split[1].split('\n')[-1].strip()
    
    messages = [{"role": "user", "content": user}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # 인퍼런스 실행
    generated_text = pipe(
        inputs,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.0001,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=terminators,
        repetition_penalty=1.5,
        )[0]['generated_text']
    
    # 문장 비교
    print('\n 입력 질문:', inputs, '\n\n 튜닝전모델 답변:', generated_text)
    if generated_text in assistant_gt or assistant_gt in generated_text:
        correct_count += 1
        print(f'Memorized: {user}{assistant_gt}')
    
    full_sentence = user + ' ' + assistant_gt # Original Training Data
    generated_full_sentence = user + ' ' +  generated_text # Generated Data with Input
    
    # 결과 저장
    data.at[index, 'generated_full_sentence'] = generated_full_sentence
    data.at[index, 'full_sentence'] = full_sentence
    print(index, 'checked.')

data.drop(columns=[header_name], inplace=True)

# 결과 출력
accuracy = correct_count / total_count
print(f"# of memorized data: {correct_count}")
print(f"Ratio of memorized data: {accuracy:.2f}")

# 최종 결과를 CSV 파일로 저장
if '1000' in file_path:
    output_path = 'Generated_1000_1ep_DP_' + file_path
else:
    output_path = 'Generated_' + file_path
data.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")