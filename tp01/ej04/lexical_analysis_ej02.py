import os
import re
import argparse
import nltk
from nltk.stem import SnowballStemmer

nltk.download('punkt')
ss = SnowballStemmer('spanish')


OUTPUT_DIR = "output"


def lex(dir: str, stop_words_file: str | None = None, min_term_length: int = 3, max_term_length: int = 23) -> None:
# len(electroencefalografista) = 23
    """
    Scan a directory recursively for txt files and return a list of all words terms found and their document frequency, plus some other statistics.
    """

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

    for root, _, files in os.walk(dir):
        for file in files:
            if re.search(r"\.txt$", file):
                try:
                    # uso de id interno para documentos
                    # genera problemas si hay nombres de archivos repetidos
                    if file in seen_files:
                        raise Exception(f"Archivo '{file}' ya procesado, no se permiten nombres de archivos repetidos.")
                    seen_files.add(file)
                    doc_id = len(doc_index)
                    doc_index[doc_id] = file
                    
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        words = re.findall(r'[a-z]+', f.read().lower())
                        for word in words:
                            if word not in stop_words and min_term_length <= len(word) <= max_term_length:
                                word = ss.stem(word)
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(f"{OUTPUT_DIR}/terminos.txt", 'w', encoding='utf-8') as f:
        for term, doc in terms.items():
            f.write(f"{term} {cfs[term]} {dfs[term]}\n")
    
    with open(f"{OUTPUT_DIR}/estadisticas.txt", 'w', encoding='utf-8') as f:    
        # cant docs procesados
        f.write(f"{len(docs)}\n")
        # cant de tokens y terminos
        cf_total = sum(cfs.values())
        f.write(f"{cf_total}\n")
        # promedio de tokens y terminos de los documentos
        f.write(f"{cf_total / len(docs) if docs else 0}\n")
        # largo promedio de un termino
        f.write(f"{sum(len(term) for term in terms) / len(terms)}\n")
        # cantidad de tokens y terminos del documento mas corto y mas largo (en cantidad de tokens)
        f.write(f"{shortest_tokens} {longest_tokens}\n")
        # cantidad de terminos que aparecen solo 1 vez en la coleccion
        f.write(f"{sum(1 for freq in cfs.values() if freq == 1)}\n")

    with open(f"{OUTPUT_DIR}/frecuencias.txt", 'w', encoding='utf-8') as f:
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

    print(f"Resultados en /{OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with the .txt files to process (scans recursively)")
    parser.add_argument("-sw", "--stop-words", help="File with stop words to ignore (separated by spaces)", default=None)
    args = parser.parse_args()

    lex(args.dir, args.stop_words)

if __name__ == "__main__":
    main()
