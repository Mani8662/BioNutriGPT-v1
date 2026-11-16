import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

t0 = time.perf_counter()
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb,
    device_map="cuda:0"
)
t_load = time.perf_counter() - t0
print(f"Model loaded in {t_load:.2f}s")

prompt = (
    "You are a clinical chrono-nutrition scheduler. Output valid JSON only.\n"
    "Allocate these Stage 3 dishes into Breakfast, Lunch, Dinner: "
    "['recipe_whole_wheat_phulka: 2 servings', 'recipe_toor_dal_tadka: 1 serving', 'recipe_palak_sabzi: 1 serving'].\n"
    "Preserve exact quantities. Do not invent new foods.\n"
    "JSON format: {\"meal_schedule\": [{\"meal\": \"Breakfast\", \"items\": [{\"food_id\": \"...\", \"quantity\": 1}]}]}"
)
inputs = tok(prompt, return_tensors="pt").to("cuda:0")

t_gen_start = time.perf_counter()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
t_gen = time.perf_counter() - t_gen_start
tokens = len(out[0]) - len(inputs["input_ids"][0])
print(f"Generated {tokens} tokens in {t_gen:.2f}s ({tokens/t_gen:.1f} tok/s)")
print(tok.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
