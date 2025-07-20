# scripts/prepare_dataset.py

import json
from pathlib import Path
from typing import Dict

INPUT_PATH = Path("data/raw_chunks.jsonl")
OUTPUT_PATH = Path("data/fine_tune_instructions.jsonl")

def generate_instruction_uz(chunk: Dict) -> str:
    """Create an Uzbek-language instruction prompt."""
    title = chunk.get("title", "").strip()
    source_type = chunk.get("source_type", "web")

    if title:
        return f"Quyidagi {source_type} manba sarlavhasi '{title}' asosida qisqacha va aniq xulosa bering."
    return f"Quyidagi {source_type} matnni aniq va tushunarli tarzda umumlashtiring."

def main():
    if not INPUT_PATH.exists():
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    instruction_count = 0
    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                chunk = json.loads(line)
                text = chunk.get("text", "").strip()
                if not text:
                    continue
                output = {
                    "instruction": generate_instruction_uz(chunk),
                    "input": "",
                    "output": text
                }
                fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                instruction_count += 1
            except Exception as e:
                print(f"⚠️ Skipping line due to error: {e}")

    print(f"✅ Tayyor. {instruction_count} ta yozuv '{OUTPUT_PATH}' fayliga saqlandi.")

if __name__ == "__main__":
    main()