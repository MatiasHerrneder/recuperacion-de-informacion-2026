import argparse
import math
import os
import pickle
import struct
import re
from tokenizer import tokenizer
from PostingChunk import PostingChunk
import shutil


def index_bsbi(corpus_path: str, memory_limit: int = 10, output_dir: str = "output3", stop_words_path: str | None = None):
    '''
    Implementa el algoritmo BSBI para indexar un corpus de documentos
    corpus_path: ruta al directorio con el corpus
    memory_limit: numero maximo de documentos a procesar en memoria antes de escribir un bloque a disco
    output_dir: ruta al directorio de salida
    stop_words_path: ruta al archivo con las stopwords
    '''
    
    MIN_TERM_LENGTH: int = 3
    MAX_TERM_LENGTH: int = 100

    INDEX_SUBPATH: str = "index.bin"
    VOCABULARY_SUBPATH: str = "vocabulary.pkl"
    TERM_ID_SUBPATH: str = "term_index.pkl"
    DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
    IDFS_SUBPATH: str = "idf.pkl"
    DOC_NORMS_SUBPATH: str = "doc_norms.pkl"
    STOPWORDS_SUBPATH: str = "stop_words.pkl"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir)

    term_to_id = {}
    memory_counter = 0
    partial_tuples = []
    chunk_id = 0

    doc_index = {}
    seen_files = set()
    
    stop_words = set()

    if stop_words_path and os.path.isfile(stop_words_path):
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            stop_words = set(re.findall(r'\w+', f.read().lower()))
    pickle.dump(stop_words, open(os.path.join(output_dir, STOPWORDS_SUBPATH), 'wb'))

    TXT_FILE_REGEX = re.compile(r"\.txt$", re.IGNORECASE)

    if os.path.isdir(corpus_path):
        files_to_process = os.walk(corpus_path)
    else:
        if os.path.isfile(corpus_path) and TXT_FILE_REGEX.search(corpus_path):
            files_to_process = [(os.path.dirname(corpus_path), [], [os.path.basename(corpus_path)])]
        else:
            raise Exception(f"Error: La ruta '{corpus_path}' no es un directorio o archivo válido.")

    for root, _, files in files_to_process:
        for file in files:
            
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                    doc_key = os.path.basename(filepath)
                    if doc_key in seen_files:
                        raise Exception(f"Archivo '{filepath}' ya procesado, no se permiten nombres de archivos repetidos.")
                    seen_files.add(doc_key)
                    doc_id = len(doc_index)
                    doc_index[doc_id] = doc_key

                    # words = tokenizer(text, stop_words=stop_words, min_length=MIN_TERM_LENGTH, max_length=MAX_TERM_LENGTH)
                    words = tokenizer(text)
                    doc_terms = {}
                    
                    for word in words:
                        if word not in doc_terms:
                            doc_terms[word] = {}
                        doc_terms[word][doc_id] = doc_terms[word].get(doc_id, 0) + 1
                        if word not in term_to_id:
                            term_to_id[word] = len(term_to_id)
                    
                    for term, doc_freq in doc_terms.items():
                        for doc_id, freq in doc_freq.items():
                            partial_tuples.append((term_to_id[term], doc_id, freq))
                    
                    memory_counter += 1

                    if memory_counter >= memory_limit:
                        partial_tuples.sort(key=lambda x: (x[0], x[1]))
                        # partial_tuples.sort(key=lambda x: (x[0]))
                        flat = [item for tup in partial_tuples for item in tup]
                        with open(os.path.join(output_dir, "chunks", f"chunk_{chunk_id}.bin"), 'wb') as bin:
                            bin.write(struct.pack(f'>{len(flat)}I', *flat))
                        chunk_id += 1
                        partial_tuples = []
                        memory_counter = 0

            except Exception as e:
                print(f"Error al leer el archivo '{filepath}': {e}")

    if partial_tuples:
        # partial_tuples.sort(key=lambda x: (x[0]))
        partial_tuples.sort(key=lambda x: (x[0], x[1]))
        flat = [item for tup in partial_tuples for item in tup]
        with open(os.path.join(output_dir, "chunks", f"chunk_{chunk_id}.bin"), 'wb') as bin:
            bin.write(struct.pack(f'>{len(flat)}I', *flat))
    

    # MERGE
    chunk_pointers = [PostingChunk(os.path.join(output_dir, "chunks", f"chunk_{i}.bin")) for i in range(chunk_id + 1)]

    vocabulary = {}
    idfs = {}
    doc_norms = {}
    N = len(doc_index)

    with open(os.path.join(output_dir, INDEX_SUBPATH), 'wb') as index:
        for term_id_actual in sorted(term_to_id.values()):
            posting_lists = []
            for chunk in chunk_pointers:
                while chunk.term_id is not None and chunk.term_id < term_id_actual:
                    chunk.next()
                
                while chunk.term_id is not None and chunk.term_id == term_id_actual:
                    _, doc_id, freq = chunk.get_record()
                    posting_lists.append((doc_id, freq))
                    chunk.next()
                    
            if posting_lists:
                seek_actual = index.tell()

                df = len(posting_lists)
                idfs[term_id_actual] = math.log(N / df)

                for doc_id, tf in posting_lists:
                    tfidf = tf * idfs[term_id_actual]
                    doc_norms[doc_id] = doc_norms.get(doc_id, 0) + tfidf ** 2

                flat = [item for tup in posting_lists for item in tup]
                index.write(struct.pack(f'>{len(flat)}I', *flat))
                vocabulary[term_id_actual] = [seek_actual, len(posting_lists)]

    doc_norms = {doc_id: math.sqrt(norm_sq) for doc_id, norm_sq in doc_norms.items()}

    pickle.dump(vocabulary, open(os.path.join(output_dir, VOCABULARY_SUBPATH), 'wb'))
    pickle.dump(term_to_id, open(os.path.join(output_dir, TERM_ID_SUBPATH), 'wb'))
    pickle.dump(doc_index, open(os.path.join(output_dir, DOCUMENT_INDEX_SUBPATH), 'wb'))
    pickle.dump(idfs, open(os.path.join(output_dir, IDFS_SUBPATH), 'wb'))
    pickle.dump(doc_norms, open(os.path.join(output_dir, DOC_NORMS_SUBPATH), 'wb'))

    # implementar que pueda retomar si se cae ??



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path", help="Directory with the files to index (scans recursively)")
    parser.add_argument("--memory_limit", type=int, default=10, help="Memory limit for buffering documents before writing to disk")
    parser.add_argument("--output_path", default="output3", help="Output directory for index files")
    parser.add_argument("--stop_words_path", help="Path to file with stop words")
    args = parser.parse_args()

    index_bsbi(args.corpus_path, args.memory_limit, args.output_path, args.stop_words_path)


if __name__ == "__main__":
    main()