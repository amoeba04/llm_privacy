import pandas as pd
import torch
from safetensors import safe_open
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for processing data')
    parser.add_argument('--model', type=str, default="solar-privacy-merged1000/checkpoint-6032",
                        help='Path to the model')
    parser.add_argument('--file_path', type=str, default='Korean_Personal_Instruction_solar_selected1000.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, default='Generated_1000_Merged_1ep_', 
                        help='Path for the output file')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    # If output_path is not provided, generate it based on file_path
    if args.output_path is None:
        args.output_path = 'Generated_1000_Merged_1ep_' + args.file_path.split('/')[-1]
    
    return args

def fix_state_dict(state_dict):
    # _module 접두어가 붙은 키를 찾아서 접두어를 제거
    fixed_state_dict = {key.replace('_module.', ''): value for key, value in state_dict.items() if '_module.' in key}
    # 원본 state_dict에서 _module 접두어가 없는 키도 추가
    fixed_state_dict.update({key: value for key, value in state_dict.items() if '_module.' not in key})
    return fixed_state_dict

# Parse command-line arguments
args = parse_arguments()

# Use the parsed arguments
MODEL = args.model
file_path = args.file_path
output_path = args.output_path + file_path
batch_size = args.batch_size

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
    tokenizer.convert_tokens_to_ids("</s>"),
    tokenizer.convert_tokens_to_ids("<|endoftext|>")
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
# max_gt_length = 0
# gt_lengths = []

for i in range(0, len(data), batch_size):
    if 'KoCommercial' in file_path and i > 100:
        break
    batch = data.iloc[i:i+batch_size]
    
    inputs_batch = []
    full_sentences_batch = []
    
    for _, row in batch.iterrows():
        row_split = row[header_name].split(tokenizer.eos_token)
        if len(row_split) == 1:
            row_split = row[header_name].split('<|eot_id|>')
        if len(row_split) == 1:
            row_split = row[header_name].split('</s>')
        if len(row_split) == 1:
            row_split = row[header_name].split('<|endoftext|>')
        if len(row_split) == 1:
            row_split = row[header_name].split('\n\n')
        
        user = row_split[0].split('\n')[-1].strip()
        assistant_gt = row_split[1].split('\n')[-1].strip()
        
        # # 토큰화하여 길이 계산
        # gt_tokens = tokenizer.encode(assistant_gt)
        # gt_length = len(gt_tokens)
        # gt_lengths.append(gt_length)
        
        # if gt_length > max_gt_length:
        #     max_gt_length = gt_length
        # print(max_gt_length)
        messages = [{"role": "user", "content": user}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs_batch.append(inputs)
        full_sentences_batch.append((user, assistant_gt))
    print(full_sentences_batch)
    
    # 배치 인퍼런스 실행
    generated_texts = pipe(
        inputs_batch,
        do_sample=True,
        max_new_tokens=64,
        temperature=0.0001,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=terminators,
        repetition_penalty=1.5,
    )

    # 결과 처리 및 저장
    for j, (generated_text, (user, assistant_gt)) in enumerate(zip(generated_texts, full_sentences_batch)):
        generated_text = generated_text[0]['generated_text']
        
        if generated_text in assistant_gt or assistant_gt in generated_text:
            correct_count += 1
            print(f'Memorized: {user}{assistant_gt}')
        
        full_sentence = user + ' ' + assistant_gt  # Original Training Data
        generated_full_sentence = user + ' ' + generated_text  # Generated Data with Input
        
        # 결과 저장
        data.at[i+j, 'generated_full_sentence'] = generated_full_sentence
        data.at[i+j, 'full_sentence'] = full_sentence
    
    print(f"Batch {i//batch_size + 1} checked.")

data.drop(columns=[header_name], inplace=True)

# 결과 출력
accuracy = correct_count / total_count
print(f"# of memorized data: {correct_count}")
print(f"Ratio of memorized data: {accuracy:.2f}")

# 최종 결과를 CSV 파일로 저장
data.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")