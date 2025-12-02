# solution.py 
import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List
import os
import re
import hashlib


MODEL_DIR = "your_model_dir/qwen2_7b_instruct"
EMBEDDING_DIR = "your_model_dir/paraphrase_multilingual_MiniLM_L12_v2"
KB_PATH = "your_model_dir/kb.json"


print("Загрузка моделей...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True  # ← КРИТИЧНО!
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    local_files_only=True
)

embedder = SentenceTransformer(EMBEDDING_DIR, device="cuda")

# KB + FAISS 
print("Загрузка KB...")
with open(KB_PATH, 'r', encoding='utf-8') as f:
    kb_data = json.load(f)

questions_kb = [item["question"] for item in kb_data]
answers_kb = [item["answer"] for item in kb_data]

# Нормализация + эмбеддинги
q_embs = embedder.encode(
    questions_kb,
    batch_size=64,
    convert_to_tensor=False,
    normalize_embeddings=True  # ← ВАЖНО!
)
q_embs = np.array(q_embs).astype('float32')

index = faiss.IndexFlatIP(q_embs.shape[1])
index.add(q_embs)

# УТИЛИТЫ
def get_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def normalize(q: str) -> str:
    return re.sub(r'[^\w\s]', '', q.lower()).strip()

# RAG 
def search_kb(question: str, threshold: float = 0.78) -> str | None:
    q_norm = normalize(question)
    q_emb = embedder.encode([q_norm], normalize_embeddings=True).astype('float32')
    D, I = index.search(q_emb, 1)
    if D[0][0] >= threshold:
        return answers_kb[I[0][0]]
    return None

#  ПРОВОКАЦИИ 
def is_provocation(q: str) -> bool:
    q = q.lower()
    return any([
        "античн" in q and any(x in q for x in ["интернет", "смартфон", "компьютер", "робот", "двигатель", "электричество"]),
        "древн" in q and "электричеств" in q,
        "до н.э." in q and "202" in q,
        "античный" in q and "xx" in q,
        "античный" in q and "век" in q and any(x in q for x in [
            "дизель", "двигатель", "радио", "телевизор", "лазер", "квантовая", "ядерная", "атомная",
            "кибернетика", "искусственный интеллект", "робот", "интернет", "спутник", "ракета"
        ]),
    ])

#  ГЕНЕРАЦИЯ 
def generate_answer(question: str) -> str:
    prompt = f"Вопрос: {question}\nОтветь кратко. Если не уверен — 'не знаю'.\nОтвет:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # ← Увеличено
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Ответ:")[-1].strip()
    answer = re.sub(r'^["\']|["\']$', '', answer)  # Убираем кавычки
    return answer if len(answer) >= 3 else "не знаю"

#  ГРУППА
def process_group(questions: List[str]) -> List[str]:
    group_key = " | ".join([normalize(q)[:40] for q in questions])
    group_hash = get_hash(group_key)
    if group_hash in group_cache:
        return group_cache[group_hash]

    # 1. Провокация
    if any(is_provocation(q) for q in questions):
     ans = ["не могу ответить на вопрос"] * len(questions)
        group_cache[group_hash] = ans
        return ans

    # 2. RAG
    kb_answer = next((search_kb(q) for q in questions), None)
    if kb_answer:
        ans = [kb_answer] * len(questions)
        group_cache[group_hash] = ans
        return ans

    # 3. Генерация
    raw = generate_answer(questions[0])
    if "не знаю" in raw.lower():
        ans = ["не знаю"] * len(questions)
    else:
        # Проверка схожести
        q_embs = embedder.encode([normalize(q) for q in questions], normalize_embeddings=True)
        sims = [cos_sim(q_embs[0], q_embs[i]).item() for i in range(len(q_embs))]
        if all(s > 0.75 for s in sims):
            ans = [raw] * len(questions)
        else:
            ans = ["не знаю"] * len(questions)
    
    group_cache[group_hash] = ans
    return ans

#  MAIN 
group_cache = {}

def main():
    with open("input.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data if isinstance(data, list) else data.get("questions", [])

    answers = []
    i = 0
    while i < len(questions):
        q = questions[i]
        group = [q]
        if i + 1 < len(questions):
            sim = cos_sim(
                embedder.encode(normalize(q), normalize_embeddings=True),
                embedder.encode(normalize(questions[i+1]), normalize_embeddings=True)
            ).item()
            if sim > 0.8:
                group.append(questions[i+1])
                if i + 2 < len(questions):
                    sim2 = cos_sim(
                        embedder.encode(normalize(q), normalize_embeddings=True),
                        embedder.encode(normalize(questions[i+2]), normalize_embeddings=True)
                    ).item()
                    if sim2 > 0.8:
                        group.append(questions[i+2])
                        i += 3
                    else:
                        i += 2
                else:
                    i += 2
            else:
                i += 1
        else:
            i += 1

        answers.extend(process_group(group))

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"Обработано {len(questions)} → {len(answers)} ответов")

if __name__ == "__main__":
    main()
