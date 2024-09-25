import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# Load pretrained model
model_name_or_path = '/mnt/sdb/privacy_backup/recent/llama3-privacy-merged1000-select-qkvo/checkpoint-3675/'  # Update with your model path or identifier
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


# Convert model to bfloat16
model = model.to(dtype=torch.bfloat16)

# Save the model
output_dir = '/mnt/sdb/privacy_backup/recent/llama3-privacy-merged1000-select-qkvo/checkpoint-3675/'  # Define the directory to save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
