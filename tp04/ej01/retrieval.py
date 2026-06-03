import argparse
import os
import pickle
import struct

INDEX_SUBPATH: str = "index.bin"
VOCABULARY_SUBPATH: str = "vocabulary.pkl"
DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
TERM_ID_SUBPATH: str = "term_index.pkl"


def retrieve_posting(index_path, term, vocabulary, term_to_id):
    if term not in term_to_id:
        return [], []

    term_id = term_to_id[term]
    seek, length = vocabulary[term_id]

    with open(os.path.join(index_path, INDEX_SUBPATH), 'rb') as f:
        f.seek(seek)
        data = f.read(length * 8)
        doc_ids, freqs = [], []
        for i in range(0, len(data), 8):
            doc_id, freq = struct.unpack('>2I', data[i:i+8])
            doc_ids.append(doc_id)
            freqs.append(freq)

    return doc_ids, freqs

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

    docs, freq = retrieve_posting(args.index_path, args.term, vocabulary, term_to_id)
    doc_names = [doc_index[doc_id] for doc_id in docs]
    for name, id, f in zip(doc_names, docs, freq):
        print(f"{name}:{id}:{f}")

if __name__ == "__main__":
    main()