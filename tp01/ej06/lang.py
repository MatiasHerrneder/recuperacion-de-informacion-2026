import os
import argparse
import math
from lang_bigrams import detect_bigram, train_bigram
from lang_langdetect import detect_langdetect


OUTPUT_DIR = "output"


def train(dir: str) -> dict[str, dict[str, float]]:
    if os.path.isdir(dir):
        files_to_process = os.walk(dir)
    else:
        if os.path.isfile(dir):
            files_to_process = [(os.path.dirname(dir), [], [os.path.basename(dir)])]
        else:
            raise Exception(f"Error: La ruta '{dir}' no es un directorio o archivo válido.")

    lang_count = {}

    for root, _, files in files_to_process:
        for file in files:
            try:
                with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                    # asumimos nombre de archivo = lenguaje
                    lang = os.path.splitext(file)[0]
                    lang_count[lang] = {}

                    for char in f.read().lower():
                        if char.isalpha():
                            lang_count[lang][char] = lang_count[lang][char] + 1 if char in lang_count[lang] else 1

            except Exception as e:
                print(f"Error al leer el archivo '{file}': {e}")

    lang_freq = {}

    for lang, counts in lang_count.items():
        total = sum(counts.values())

        if total == 0:
            lang_freq[lang] = {}
        else:
            lang_freq[lang] = {
                char: count / total
                for char, count in counts.items()
            }
    return lang_freq
    

def euclidean_distance(freq1: dict[str, float], freq2: dict[str, float]) -> float:
    letters = set(freq1.keys()) | set(freq2.keys())
    return math.sqrt(sum((freq1.get(l, 0) - freq2.get(l, 0))**2 for l in letters))


def detect(file: str, lang_freq: dict[str, dict[str, float]]) -> list[tuple[int, str]]:
    if not os.path.isfile(file):
        raise Exception(f"Error: La ruta '{file}' no es un archivo válido.")

    results: list[tuple[int, str]] = []

    try:
        with open(file, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f, start=1):
                counts: dict[str, int] = {}

                for char in line.lower():
                    if char.isalpha():
                        counts[char] = counts.get(char, 0) + 1

                total = sum(counts.values())

                if total == 0:
                    results.append((i, "unknown"))
                    continue

                doc_freq = {
                    char: count / total
                    for char, count in counts.items()
                }

                distances: dict[str, float] = {}
                for lang, freq in lang_freq.items():
                    distances[lang] = euclidean_distance(doc_freq, freq)

                detected = min(distances, key=lambda k: distances[k])
                results.append((i, detected))

    except Exception as e:
        print(f"Error al leer el archivo '{file}': {e}")

    return results


def save_results(results: list[tuple[int, str]], output_dir: str) -> None:
    output_file = os.path.join(output_dir, "results.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for line_num, lang in results:
            f.write(f"{line_num} {lang}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="Directory with the files to process")
    parser.add_argument("detection", help="File to detect")
    args = parser.parse_args()

    prob_matrix = train_bigram(args.training)
    results_b = detect_bigram(args.detection, prob_matrix)
    
    lang_model = train(args.training)
    results = detect(args.detection, lang_model)

    results_l = detect_langdetect(args.detection)

    save_results(results, os.path.join(OUTPUT_DIR, "results.txt"))
    save_results(results_b, os.path.join(OUTPUT_DIR, "results_bigram.txt"))
    save_results(results_l, os.path.join(OUTPUT_DIR, "results_langdetect.txt"))


    print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()