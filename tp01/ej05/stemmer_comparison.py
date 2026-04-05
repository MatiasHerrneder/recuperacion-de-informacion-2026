import os
import re
import argparse
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer
import time
from enum import Enum

nltk.download('punkt')


OUTPUT_DIR = "output"


class StemmerType(Enum):
    PORTER = "porter"
    LANCASTER = "lancaster"

def tokenize(text: str, stemming: StemmerType | None = None) -> list[str]:
    """
    Implements the tokenization logic.
    """

    EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    emails = EMAIL_REGEX.findall(text)
    remaining = EMAIL_REGEX.sub("", text)

    # URL_REGEX = re.compile(r'[a-zA-Z]+://[^\s<>"\']+(?<![.,;:!?)\'"])')
    URL_REGEX = re.compile(r'[a-zA-Z]+://[-a-zA-Z0-9@:%._\+~#=]+(?:\.[a-zA-Z0-9()]+)+(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    urls = URL_REGEX.findall(remaining)
    remaining = URL_REGEX.sub("", remaining)

    DATE_REGEX = re.compile(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b')
    dates = DATE_REGEX.findall(remaining)
    remaining = DATE_REGEX.sub("", remaining)

    PHONE_REGEX = re.compile(r'(?:\+\d{1,3}[ -])?\(?\d{2,4}\)?[ -]\d{3,4}(?:[ -]\d{3,4})?')
    phones = PHONE_REGEX.findall(remaining)
    remaining = PHONE_REGEX.sub("", remaining)

    NUMBER_REGEX = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
    numbers = NUMBER_REGEX.findall(remaining)
    remaining = NUMBER_REGEX.sub("", remaining)

    ABREVIATURES_REGEX = re.compile(r'\b(?:[A-Z][a-z]*\.)+')
    abrevitures = ABREVIATURES_REGEX.findall(remaining)
    remaining = ABREVIATURES_REGEX.sub("", remaining)

    ACRONYM_REGEX = re.compile(r'\b[A-Z]{2,}\b')
    acronyms = ACRONYM_REGEX.findall(remaining)
    remaining = ACRONYM_REGEX.sub("", remaining)

    PROPER_NAME_REGEX = re.compile(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b')
    names = PROPER_NAME_REGEX.findall(remaining)
    remaining = PROPER_NAME_REGEX.sub("", remaining)

    remaining = re.findall(r'[^\W\d_]+', remaining.lower())
    
    if stemming == StemmerType.PORTER:
        for word in remaining:
            word = porter_stemmer(word)
    elif stemming == StemmerType.LANCASTER:
        for word in remaining:
            word = lancaster_stemmer(word)

    return emails + urls + dates + phones + numbers + abrevitures + acronyms + names + remaining


def tokenizer(dir: str, stop_words_file: str | None = None, stemming: StemmerType | None = None, min_term_length: int = 3, max_term_length: int = 300) -> None:
    # len(electroencefalografista) = 23
    """
    Scan a directory recursively for txt files and return a list of all words terms found and their document frequency, plus some other statistics.
    """

    LOCAL_OUTPUT = f"{OUTPUT_DIR}/{stemming.value}" if stemming else f"{OUTPUT_DIR}/no_stemming"

    terms = {}
    stop_words = set()
    doc_index = {}
    seen_files = set()

    if stop_words_file and os.path.isfile(stop_words_file):
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            stop_words = set(re.findall(r'[a-z]+', f.read().lower()))
    
    if not os.path.isdir(dir):
        print(f"Error: La ruta '{dir}' no es un directorio válido.")
        return

    TXT_FILE_REGEX = re.compile(r"\.txt$", re.IGNORECASE)

    for root, _, files in os.walk(dir):
        for file in files:
            if TXT_FILE_REGEX.search(file):
                try:
                    # uso de id interno para documentos
                    # genera problemas si hay nombres de archivos repetidos
                    if file in seen_files:
                        raise Exception(f"Archivo '{file}' ya procesado, no se permiten nombres de archivos repetidos.")
                    seen_files.add(file)
                    doc_id = len(doc_index)
                    doc_index[doc_id] = file
                    
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        words = tokenize(f.read(), stemming)
                        for word in words:
                            if word not in stop_words and min_term_length <= len(word) <= max_term_length:
                                if word not in terms:
                                    terms[word] = {}
                                terms[word][doc_id] = terms[word].get(doc_id, 0) + 1
                except Exception as e:
                    print(f"Error al leer el archivo '{file}': {e}")
                
    terms = dict(sorted(terms.items()))

    # para el calculo de estadisticas extraigo otras estructuras
    dfs = {}
    cfs = {}
    docs = {}
    for term, doc_list in terms.items():
        dfs[term] = len(doc_list)
        cfs[term] = sum(doc_list.values())
        for doc, freq in doc_list.items():
            if doc not in docs:
                docs[doc] = {}
            docs[doc][term] = freq
    cfs = dict(sorted(cfs.items(), key=lambda item: item[1]))
    
    if not docs:
        shortest_tokens = longest_tokens = 0
    else:
        _, shortest_terms = min(
            docs.items(),
            key=lambda item: sum(item[1].values())
        )
        _, longest_terms = max(
            docs.items(),
            key=lambda item: sum(item[1].values())
        )
        shortest_tokens = sum(shortest_terms.values())
        longest_tokens = sum(longest_terms.values())

    os.makedirs(LOCAL_OUTPUT, exist_ok=True)

    with open(f"{LOCAL_OUTPUT}/terminos.txt", 'w', encoding='utf-8') as f:
        for term, doc in terms.items():
            f.write(f"{term} {cfs[term]} {dfs[term]}\n")
    
    with open(f"{LOCAL_OUTPUT}/estadisticas.txt", 'w', encoding='utf-8') as f:    
        # cant docs procesados
        f.write(f"{len(docs)}\n")
        # cant de tokens y terminos
        cf_total = sum(cfs.values())
        f.write(f"{cf_total}\n")
        # promedio de tokens y terminos de los documentos
        f.write(f"{cf_total / len(docs) if docs else 0}\n")
        # largo promedio de un termino
        if len(terms) > 0:
            f.write(f"{sum(len(term) for term in terms) / len(terms)}\n")
        else:
            f.write("0\n")
        # cantidad de tokens y terminos del documento mas corto y mas largo (en cantidad de tokens)
        f.write(f"{shortest_tokens} {longest_tokens}\n")
        # cantidad de terminos que aparecen solo 1 vez en la coleccion
        f.write(f"{sum(1 for freq in cfs.values() if freq == 1)}\n")

    with open(f"{LOCAL_OUTPUT}/frecuencias.txt", 'w', encoding='utf-8') as f:
        count = 0
        for term, freq in reversed(cfs.items()):
            if count == 10:
                break
            f.write(f"{term} {freq}\n")
            count += 1
        count = 0
        for term, freq in cfs.items():
            if count == 10:
                break
            f.write(f"{term} {freq}\n")
            count += 1

    print(f"Resultados en /{LOCAL_OUTPUT}")


def porter_stemmer(text: str | list[str]) -> set[str]:
    return stemmer(PorterStemmer(), text)

def lancaster_stemmer(text: str | list[str]) -> set[str]:
    return stemmer(LancasterStemmer(), text)

def stemmer(stemmer_mod, text: str | list[str]) -> set[str]:
    if isinstance(text, str):
        return {stemmer_mod.stem(text)}
    return {stemmer_mod.stem(word) for word in text}


def compare(dir: str, stop_words_file: str | None = None) -> None:

    #TODO falta preprocesado 

    start = time.perf_counter()
    tokenizer(dir, stop_words_file, StemmerType.PORTER)
    end = time.perf_counter()
    porter_time = end - start
    
    start = time.perf_counter()
    tokenizer(dir, stop_words_file, StemmerType.LANCASTER)
    end = time.perf_counter()
    lancaster_time = end - start

    with open(f"{OUTPUT_DIR}/times.txt", 'w', encoding='utf-8') as f:
        f.write(f"Porter Stemmer took: {porter_time:4f} seconds\n")
        f.write(f"Lancaster Stemmer took: {lancaster_time:.4f} seconds\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with the .txt files to process (scans recursively)")
    parser.add_argument("-sw", "--stop-words", help="File with stop words to ignore (separated by spaces)", default=None)
    args = parser.parse_args()

    compare(args.dir, args.stop_words)

if __name__ == "__main__":
    main()