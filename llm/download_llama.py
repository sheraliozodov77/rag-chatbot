# llm/download_llama.py

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Downloading model weights...")
model = AutoModelForCausalLM.from_pretrained(model_id)

print("Model downloaded and cached locally.")
