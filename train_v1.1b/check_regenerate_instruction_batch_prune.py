import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from peft import AutoPeftModelForCausalLM
import argparse
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for processing data')
    parser.add_argument('--model', type=str, default="eeve-lora-privacy-merged1000",
                        help='Path to the model')
    parser.add_argument('--file_path', type=str, default='Korean_Personal_Instruction_eeve_selected1000.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, default='Generated_1000_KoCommercial_3ep_LoRA_', 
                        help='Path for the output file')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--prune_ratio', type=float, default=0.5,
                        help='Pruning ratio')
    
    args = parser.parse_args()
    
    # If output_path is not provided, generate it based on file_path
    if args.output_path is None:
        args.output_path = 'Generated_1000_Merged_1ep_' + args.file_path.split('/')[-1]
    
    return args

def preprocess_function(examples):
    row_split = examples[header_name].split(tokenizer.eos_token)
    if len(row_split) == 1:
        row_split = examples[header_name].split('<|eot_id|>')
    if len(row_split) == 1:
        row_split = examples[header_name].split('</s>')
    if len(row_split) == 1:
        row_split = examples[header_name].split('<|endoftext|>')
    if len(row_split) == 1:
        row_split = examples[header_name].split('\n\n')
    if len(row_split) == 1:
        row_split = examples[header_name].split('<|im_end|>\n')
    
    user = row_split[0].split('\n')[-1].strip()
    assistant_gt = row_split[1].split('\n')[-1].strip()
    
    messages = [{"role": "user", "content": user}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', pad_to_max_length=True)
    model_inputs["labels"] = tokenizer(assistant_gt, max_length=64, truncation=True, padding='max_length', pad_to_max_length=True)["input_ids"]
    
    return model_inputs

# Parse command-line arguments
args = parse_arguments()

MODEL = args.model
file_path = args.file_path
output_path = args.output_path + os.path.basename(file_path)
batch_size = args.batch_size

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=args.prune_ratio)
        prune.remove(module, 'weight')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if 'llama3' in MODEL:
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        # tokenizer.convert_tokens_to_ids("</s>"),
        # tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        # tokenizer.convert_tokens_to_ids("\n\n"),
    ]
else:
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("</s>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        tokenizer.convert_tokens_to_ids("\n\n"),
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

# Dataset 생성
dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

# DataCollator 설정
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=128, return_tensors="pt")

# DataLoader 생성
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

correct_count = 0
total_count = len(data)
results = []

model.eval()

for batch in dataloader:
    with torch.no_grad():
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=terminators,
            repetition_penalty=1.5,
        )
    
    inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ground_truth = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    
    for inp, generated_text, gt in zip(inputs, generated_texts, ground_truth):
        generated_text = generated_text.replace(inp, "")
        if generated_text in gt or gt in generated_text:
            correct_count += 1
            print(f'Memorized: {gt}')
        
        # 결과를 리스트에 추가
        results.append({
            'generated_full_sentence': inp+generated_text,
            'full_sentence': inp+gt
        })
        print(results[-1])

    print(f"Batch processed.")

# 결과 출력
accuracy = correct_count / total_count
print(f"# of memorized data: {correct_count}")
print(f"Ratio of memorized data: {accuracy:.2f}")

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 최종 결과를 CSV 파일로 저장
results_df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")