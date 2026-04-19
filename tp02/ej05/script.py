import math
import os
import re
import argparse
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')


OUTPUT_DIR = "output"


def tokenizer(text: str) -> list[str]:
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

    stemmer = PorterStemmer()
    remaining = [stemmer.stem(term) for term in remaining]

    return emails + urls + dates + phones + numbers + abrevitures + acronyms + names + remaining


def indexer(dir: str, stop_words_file: str | None = None, min_term_length: int = 3, max_term_length: int = 100):
    # len(electroencefalografista) = 23
    """
    Returns an inverted index in memory of the documents in the given directory.
    """

    terms = {}
    stop_words = set()
    doc_index = {}
    seen_files = set()

    if stop_words_file and os.path.isfile(stop_words_file):
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            stop_words = set(re.findall(r'[a-z]+', f.read().lower()))
    
    TXT_FILE_REGEX = re.compile(r"\.txt$", re.IGNORECASE)

    if os.path.isdir(dir):
        files_to_process = os.walk(dir)
    else:
        if os.path.isfile(dir) and TXT_FILE_REGEX.search(dir):
            files_to_process = [(os.path.dirname(dir), [], [os.path.basename(dir)])]
        else:
            raise Exception(f"Error: La ruta '{dir}' no es un directorio o archivo válido.")

    for root, _, files in files_to_process:
        for file in files:
            # if TXT_FILE_REGEX.search(file): 
            try:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    raw_docs = [(file, content)]

                    for docno, text in raw_docs:
                        doc_key = f"{file}::{docno}" if docno else file
                        if doc_key in seen_files:
                            raise Exception(f"Archivo '{file}' ya procesado, no se permiten nombres de archivos repetidos.")
                        seen_files.add(doc_key)
                        doc_id = len(doc_index)
                        doc_index[doc_id] = doc_key

                        words = tokenizer(text)
                        
                        for word in words:
                            if word not in stop_words and min_term_length <= len(word) <= max_term_length:
                                if word not in terms:
                                    terms[word] = {}
                                terms[word][doc_id] = terms[word].get(doc_id, 0) + 1
            except Exception as e:
                print(f"Error al leer el archivo '{file}': {e}")

    doc_norms = [0.0] * len(doc_index)
    for term, postings in terms.items():
        idf = math.log(len(doc_index) / len(postings))
        for doc_id, freq in postings.items():
            doc_norms[doc_id] += ((1 + math.log(freq)) * idf) ** 2

    doc_norms = [math.sqrt(n) for n in doc_norms]
                
    return dict(inverted_index=dict(sorted(terms.items())), corpus_size=len(doc_index), doc_index=doc_index, doc_norms=doc_norms)

def frequencies(l):
    count = {}
    for elem in l:
        count[elem] = count.get(elem, 0) + 1
    return list(count.items())

def retriever(index, corpus_size, query, doc_norms) -> list[tuple[int, float]]:
    terms = frequencies([t for t in tokenizer(query) if 3 <= len(t) <= 100])
    
    query_vector = [0.0] * len(terms)
    doc_vectors = {}
    for i, (term, value) in enumerate(terms):
        if term not in index:
            continue
        idf = (math.log(corpus_size / len(index[term])))
        query_vector[i] = (1 + math.log(value)) * idf
        
        if term in index:
            for doc_id, freq in index[term].items():
                if doc_id not in doc_vectors:
                    doc_vectors[doc_id] = [0] * len(terms)
                doc_vectors[doc_id][i] = (1 + math.log(freq)) * idf

    similarities = []
    for doc_id, doc_vector in doc_vectors.items():
        sim = cosine_similarity(query_vector, doc_vector, doc_norms[doc_id])
        similarities.append((doc_id, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def cosine_similarity(vec1, vec2, doc_norm=None):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = doc_norm if doc_norm is not None else math.sqrt(sum(b ** 2 for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with the files to index (scans recursively)")
    parser.add_argument("queries", help="Queries to search (separated by commas)")
    args = parser.parse_args()

    result = indexer(args.dir)
    index = result["inverted_index"]
    corpus_size = result["corpus_size"]
    doc_index = result["doc_index"]
    doc_norms = result["doc_norms"]

    results = {}
    if index is not None:
        queries = args.queries.split(",")
        for query in queries:
            results[query] = retriever(index, corpus_size, query, doc_norms)

    assert type(doc_index) == dict
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, (query, res) in enumerate(results.items()):
        with open(os.path.join(OUTPUT_DIR, f"Q{i}.txt"), 'w', encoding='utf-8') as f:
            for doc_id, sim in res:
                f.write(f"{doc_index[doc_id]} : {sim}\n")

if __name__ == "__main__":
    main()