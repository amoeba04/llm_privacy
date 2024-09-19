import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# Load pretrained model
# model_name_or_path = 'solar-privacy-merged1000-noise0.025/checkpoint-6032'  # Update with your model path or identifier
model_name_or_path = '/mnt/sdb/privacy_backup/recent/eeve-privacy-merged1000-noise0.01/checkpoint-3475'
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


# Convert model to bfloat16
model = model.to(dtype=torch.bfloat16)

# Save the model
# output_dir = 'solar-privacy-merged1000-noise0.025/checkpoint-6032'  # Define the directory to save the model
output_dir = '/mnt/sdb/privacy_backup/recent/eeve-privacy-merged1000-noise0.01/checkpoint-3475'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
