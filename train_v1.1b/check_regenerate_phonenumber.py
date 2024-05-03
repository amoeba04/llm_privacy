import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# MODEL = "/home/privacy/KoAlpaca/train_v1.1b/eeve-10.8b-privacy-sentence"
# MODEL = "/home/privacy/KoAlpaca/train_v1.1b/eeve-10.8b-privacy-sentence-dedupname"
# MODEL = "/home/privacy/KoAlpaca/train_v1.1b/eeve-10.8b-privacy-phonenumber-dedupname-redup-level10"
MODEL = "/home/privacy/KoAlpaca/train_v1.1b/llama3-8b-privacy-phonenumber-dedupname-redup-level100"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
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
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

file_path = '/home/privacy/KoAlpaca/train_v1.1b/Generated_Korean_Phonenumber_dedupname_edit.csv'

data = pd.read_csv(file_path, header=None, names=['text'])
data['text'] = data['text'].astype(str)

# 문장 생성 및 검증
correct_count = 0
total_count = len(data)

for index, row in data.iterrows():
    first_word = ' '.join(row['text'].split()[:2])  # 첫 단어 추출
    full_sentence = row['text'].replace('<|im_start|>user\n', '').replace('<|im_start|>user ', '').replace('<|im_end|>', '')
    full_sentence = full_sentence.replace('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n', '').replace('<|begin_of_text|><|start_header_id|>user<|end_header_id|> ', '').replace('<|eot_id|>', '')
    full_sentence = full_sentence.strip()

    # 인퍼런스 실행
    generated_text = pipe(
        first_word,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.0001,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=terminators
        )
    generated_full_sentence = first_word + generated_text[0]['generated_text']
    generated_full_sentence = generated_full_sentence.replace('<|im_start|>user\n', '').replace('<|im_start|>user ', '').replace('<|im_end|>', '')
    generated_full_sentence = generated_full_sentence.replace('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n', '').replace('<|begin_of_text|><|start_header_id|>user<|end_header_id|> ', '').replace('<|eot_id|>', '')
    generated_full_sentence = generated_full_sentence.strip()

    # 문장 비교
    if generated_full_sentence == full_sentence:
        correct_count += 1
        print('MATCHED!')
    
    # 결과 저장
    data.at[index, 'generated_full_sentence'] = generated_full_sentence
    data.at[index, 'full_sentence'] = full_sentence
    print(index, 'checked.')

data.drop(columns=['text'], inplace=True)

# 결과 출력
accuracy = correct_count / total_count
print(f"Correctly matched sentences: {correct_count}")
print(f"Accuracy: {accuracy:.2f}")

# 최종 결과를 CSV 파일로 저장
output_path = '/home/privacy/KoAlpaca/train_v1.1b/Generated_Korean_Phonenumber_dedupname_edit_redup_level100_comparison_llama3.csv'
data.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")