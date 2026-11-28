import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-3B-Instruct"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb,
    device_map="cuda:0",
)

basket = [
    {"food_id": "recipe_whole_wheat_roti", "name": "Whole Wheat Roti", "quantity": 2},
    {"food_id": "recipe_yellow_moong_dal", "name": "Yellow Moong Dal", "quantity": 1},
    {"food_id": "recipe_bhindi_masala", "name": "Bhindi Masala", "quantity": 1},
    {"food_id": "ifct_f028", "name": "Guava", "quantity": 1},
    {"food_id": "recipe_curd_dahi_low_fat", "name": "Curd", "quantity": 1},
]

prompt = (
    f"You are the BioNutriGPT Stage 4 Clinical Compiler.\n"
    f"Stage 3 Basket: {json.dumps(basket)}\n\n"
    f"Instructions:\n"
    f"1. Allocate EVERY food into 5 meal slots: Breakfast, Mid-Morning Snack, Lunch, Evening Snack, Dinner.\n"
    f"2. Strict Food & Quantity Conservation: Every dish must appear with its exact quantity preserved across meals. Do not add or drop items.\n"
    f"3. Strict Arithmetic Rule: Do NOT mention any numerical calorie, protein, or nutrient quantities in narrative.\n"
    f"4. Keep narrative under 10 words. Output single-line compact JSON only conforming to:\n"
    f'{{"narrative":"...","meals":{{"Breakfast":[{{"food_id":"...","quantity":1}}],"Mid-Morning Snack":[],"Lunch":[],"Evening Snack":[],"Dinner":[]}}}}'
)

messages = [
    {"role": "system", "content": "You are the BioNutriGPT Stage 4 Clinical Compiler. Output valid compact JSON only."},
    {"role": "user", "content": prompt}
]
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")

t1 = time.perf_counter()
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=110,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or 151643,
        eos_token_id=[151643, 151645],
    )
gen_time = time.perf_counter() - t1
gen_tokens = output[0][inputs["input_ids"].shape[1]:]
print(f"Generated {len(gen_tokens)} tokens in {gen_time:.2f}s ({len(gen_tokens)/gen_time:.1f} tok/s)")
text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
print("Output text:")
print(text)

try:
    clean_text = text.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(clean_text)
    print("\nJSON Parse OK! Keys:", list(data.keys()))
    print("Meals:", list(data.get("meals", {}).keys()))
except Exception as e:
    print("\nJSON Parse Error:", e)
