import os

import math


BigramModel = dict[str, dict[str, float]]

def train_bigram(dir: str) -> dict[str, BigramModel]:
    lang_counts: dict[str, dict[str, dict[str, int]]] = {}
    totals: dict[str, dict[str, int]] = {}

    for root, _, files in os.walk(dir):
        for file in files:
            try:
                with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                    lang = os.path.splitext(file)[0]

                    lang_counts[lang] = {}
                    totals[lang] = {}

                    text = "".join(c for c in f.read().lower() if c.isalpha())

                    for i in range(len(text) - 1):
                        x = text[i]
                        y = text[i + 1]

                        if x not in lang_counts[lang]:
                            lang_counts[lang][x] = {}

                        lang_counts[lang][x][y] = lang_counts[lang][x].get(y, 0) + 1
                        totals[lang][x] = totals[lang].get(x, 0) + 1

            except Exception as e:
                print(f"Error: {e}")

    model: dict[str, BigramModel] = {}

    for lang in lang_counts:
        model[lang] = {}

        for x in lang_counts[lang]:
            model[lang][x] = {}

            for y in lang_counts[lang][x]:
                model[lang][x][y] = lang_counts[lang][x][y] / totals[lang][x]
    return model

def score_text_bigram(text: str, model: BigramModel) -> float:
    clean = "".join(c for c in text.lower() if c.isalpha())

    score = 0.0

    for i in range(len(clean) - 1):
        x = clean[i]
        y = clean[i + 1]

        prob = model.get(x, {}).get(y, 1e-6)  # smoothing
        score += math.log(prob)

    return score

def detect_bigram(file: str, models: dict[str, BigramModel]) -> list[tuple[int, str]]:
    results = []

    with open(file, 'r', encoding='latin-1') as f:
        for line_num, line in enumerate(f, start=1):

            best_lang = None
            best_score = float("-inf")

            for lang, model in models.items():
                score = score_text_bigram(line, model)

                if score > best_score:
                    best_score = score
                    best_lang = lang

            results.append((line_num, best_lang if best_lang else "unknown"))

    return results