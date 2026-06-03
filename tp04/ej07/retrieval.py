import argparse
import os
import pickle

from indexer import decode_posting_list

INDEX_SUBPATH: str = "index.bin"
VOCABULARY_SUBPATH: str = "vocabulary.pkl"
DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
TERM_ID_SUBPATH: str = "term_index.pkl"


def retrieve_posting(index_path, term, vocabulary, term_to_id):
    if term not in term_to_id:
        return [], [], []

    term_id = term_to_id[term]
    seek, n_docs, n_bytes = vocabulary[term_id]

    with open(os.path.join(index_path, INDEX_SUBPATH), 'rb') as f:
        f.seek(seek)
        data = f.read(n_bytes)

    posting_list = decode_posting_list(data, n_docs)
    doc_ids = [doc_id for doc_id, _ in posting_list]
    freqs   = [freq   for _, freq   in posting_list]

    return doc_ids, freqs, data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Directory with the index files")
    parser.add_argument("term", help="Term to search for")

    args = parser.parse_args()

    with open(os.path.join(args.index_path, VOCABULARY_SUBPATH), 'rb') as f:
        vocabulary = pickle.load(f)

    with open(os.path.join(args.index_path, TERM_ID_SUBPATH), 'rb') as f:
        term_to_id = pickle.load(f)

    with open(os.path.join(args.index_path, DOCUMENT_INDEX_SUBPATH), 'rb') as f:
        doc_index = pickle.load(f)

    docs, freq, data = retrieve_posting(args.index_path, args.term, vocabulary, term_to_id)
    doc_names = [doc_index[doc_id] for doc_id in docs]
    for name, id, f in zip(doc_names, docs, freq):
        print(f"{name}:{id}:{f}")
    if isinstance(data, (bytes, bytearray)):
        print("Posting list en bytes:")
        print(data.hex())

if __name__ == "__main__":
    main()