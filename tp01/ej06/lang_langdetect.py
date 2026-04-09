from langdetect import detect, DetectorFactory, LangDetectException
import os

DetectorFactory.seed = 0

def detect_langdetect(file: str) -> list[tuple[int, str]]:
    if not os.path.isfile(file):
        raise Exception(f"Error: La ruta '{file}' no es un archivo válido.")

    results: list[tuple[int, str]] = []

    lang_map = {
        "en": "English",
        "fr": "French",
        "it": "Italian",
    }

    try:
        with open(file, 'r', encoding='latin-1') as f:
            for line_num, line in enumerate(f, start=1):
                text = line.strip()

                if not text:
                    results.append((line_num, "unknown"))
                    continue

                try:
                    lang_code = detect(text)
                    lang = lang_map.get(lang_code, lang_code)
                except LangDetectException:
                    lang = "unknown"

                results.append((line_num, lang))

    except Exception as e:
        print(f"Error al leer el archivo '{file}': {e}")

    return results