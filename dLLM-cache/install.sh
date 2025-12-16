conda create -n dllm_cache python=3.12
conda activate dllm_cache
pip install -r requirements.txt
huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Instruct
huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Base
huggingface-cli download --resume-download Dream-org/Dream-v0-Instruct-7B
huggingface-cli download --resume-download Dream-org/Dream-v0-Base-7B
