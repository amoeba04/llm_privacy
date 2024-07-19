import pandas as pd
import re
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine

# NLP 엔진 및 Analyzer, Anonymizer 설정
model_config = [{"lang_code": "en", "model_name": {
    "spacy": "en_core_web_sm",
    "transformers": "Leo97/KoELECTRA-small-v3-modu-ner"
    }
}]

nlp_engine = TransformersNlpEngine(models=model_config)
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=['ko', 'en'])
anonymizer = AnonymizerEngine()

def get_model_tokens(model_name):
    if 'eeve' in model_name.lower() or 'midm' in model_name.lower() or 'gemma' in model_name.lower():
        return {'start': '<|im_start|>', 'end': '<|im_end|>\n', 'user': 'user\n', 'bot': 'assistant\n'}
    elif 'llama3' in model_name.lower():
        return {'start': '<|begin_of_text|>', 'end': '<|eot_id|>', 'user': '<|start_header_id|>user<|end_header_id|>\n\n', 'bot': '<|start_header_id|>assistant<|end_header_id|>\n\n'}
    elif 'llama2' in model_name.lower() or 'mistral' in model_name.lower():
        return {'start': '<s>', 'end': ' </s>', 'user': '[INST] ', 'bot': ' [/INST] '}
    elif 'solar' in model_name.lower():
        return {'start': '', 'end': '\n\n', 'user': '### User:\n', 'bot': '### Assistant:\n'}
    else:
        raise ValueError('Model Chat Template Not implemented.')

def extract_assistant_response(text, tokens):
    pattern = re.escape(tokens['bot']) + r'(.*?)' + re.escape(tokens['end'])
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def anonymize_text(text):
    results = analyzer.analyze(text=text, language='en')
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_text.text

def process_file(file_path, model_name):
    tokens = get_model_tokens(model_name)
    df = pd.read_csv(file_path)
    new_rows = []

    for index, row in df.iterrows():
        original_text = row['Generated Sentence']
        assistant_response = extract_assistant_response(original_text, tokens)
        anonymized_response = anonymize_text(assistant_response)
        
        new_text = original_text.replace(assistant_response, anonymized_response)
        new_rows.append({'Generated Sentence': new_text})

    new_df = pd.DataFrame(new_rows)
    output_file = f'Korean_Personal_Instruction_{model_name}_redup_levels1000_filtered.csv'
    new_df.to_csv(output_file, index=False)
    print(f"익명화된 응답이 포함된 새 CSV 파일이 생성되었습니다: {output_file}")

# 모델 리스트와 해당 파일 경로
models_and_files = [
    ('midm', 'Korean_Personal_Instruction_midm_redup_levels1000.csv'),
]

# 각 모델에 대해 처리 실행
for model_name, file_path in models_and_files:
    process_file(file_path, model_name)