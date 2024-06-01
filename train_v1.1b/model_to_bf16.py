import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Load pretrained model
model_name_or_path = 'eeve-privacy-one/checkpoint-1246'  # Update with your model path or identifier
config = AutoConfig.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

# Convert model to bfloat16
model = model.to(dtype=torch.bfloat16)

# Save the model
output_dir = 'eeve-privacy-one/checkpoint-1246'  # Define the directory to save the model
model.save_pretrained(output_dir)
