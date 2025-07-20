# llm/llm_loader.py

import os
from dotenv import load_dotenv
from utils.logger import setup_logger

logger = setup_logger()
load_dotenv()

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_llm(model_name="gpt-4", temperature=0.1):
    if USE_LOCAL_LLM:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        logger.info(f"Loading local LLaMA model from: {LOCAL_LLM_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_PATH, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            max_new_tokens=512,
            repetition_penalty=1.1
        )
        return pipe
    else:
        from langchain.chat_models import ChatOpenAI
        logger.info("Using OpenAI GPT model via API.")
        return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
