import os
import torch
import subprocess
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_models/'

from transformers import AutoTokenizer, AutoModel

# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Model and Tokenizer
# Using the Instruct model as per chat.py example
model_id = 'GSAI-ML/LLaDA-8B-Instruct'

print(f"Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,).to(device).eval()
