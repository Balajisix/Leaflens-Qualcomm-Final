import os
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM, ORTConfig
import onnxruntime as ort

# 1) Paths
MODEL_DIR = "llama-3.2-3b-onnx-qnn"

# 2) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

# 3) ONNX Runtime config: pick your provider
providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
ort_config = ORTConfig(
    providers=providers,
    # you can tweak session options here if needed
)

# 4) Load the ONNX model for generation
model = ORTModelForCausalLM.from_pretrained(
    MODEL_DIR,
    file_name="plant_leaf_diseases_model.onnx",
    ort_config=ort_config,
)

# 5) Generation helper
def generate(prompt: str, max_new_tokens: int = 128):
    inputs = tokenizer(
        prompt,
        return_tensors="np",
        padding=False,
        truncation=True,
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_prompt = (
        "### Instruction:\n"
        "Translate the following English text to French:\n"
        "The leaves are turning yellow and curling at the edges.\n"
        "### Response:"
    )
    print(generate(sample_prompt))
