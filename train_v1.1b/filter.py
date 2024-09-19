import pandas as pd
import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.nlp_engine import TransformersNlpEngine
import warnings

warnings.filterwarnings("ignore")

# Presidio 설정
model_config = [{"lang_code": "en", "model_name": {
    "spacy": "en_core_web_sm",
    "transformers": "Leo97/KoELECTRA-small-v3-modu-ner"
    }
}]

nlp_engine = TransformersNlpEngine(models=model_config)

analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=['ko', 'en'])
anonymizer = AnonymizerEngine()

# CSV 파일 읽기
df = pd.read_csv('results_privacy/Generated_1000_Merged_3ep_Korean_Personal_Instruction_eeve_selected1000.csv')

def process_text(text):
    try:
        results = analyzer.analyze(text=text, language='en')
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_text.text
    except OverflowError:
        print(f"Error processing text: {text[:100]}...")  # 문제가 있는 텍스트의 처음 100자만 출력
        return text

# 처리된 문장을 저장할 리스트
processed_sentences = []

# 각 문장에 대해 Presidio를 사용하여 익명화 수행
for sentence in df['generated_full_sentence']:
    if '? ' in sentence:
        parts = sentence.split('? ', 1)
        separator = '? '
    else:
        parts = sentence.split('. ', 1)
        separator = '. '
    
    # 앞부분과 뒷부분 모두 처리
    processed_prefix = process_text(parts[0][:64])  # 앞부분 처리
    processed_suffix = process_text(parts[1][:64]) if len(parts) > 1 else ""  # 뒷부분 처리

    # 처리된 앞부분과 뒷부분을 다시 합침
    processed_sentence = processed_prefix + separator + processed_suffix
    processed_sentences.append(processed_sentence)

    # 진행 상황 출력 (선택사항)
    if len(processed_sentences) % 1000 == 0:
        print(f"Processed {len(processed_sentences)} sentences")

# 처리된 문장의 수와 원본 문장의 수가 다를 경우, 부족한 부분을 빈 문자열로 채움
if len(processed_sentences) < len(df):
    processed_sentences.extend([""] * (len(df) - len(processed_sentences)))

# 처리된 데이터를 데이터프레임에 다시 저장
df['generated_full_sentence'] = processed_sentences

# 결과를 CSV 파일로 저장
df.to_csv('results_filtered_presidio/InOutFilter_Generated_1000_Merged_3ep_Korean_Personal_Instruction_eeve_selected1000.csv', index=False)

print("Processing completed and results saved.")