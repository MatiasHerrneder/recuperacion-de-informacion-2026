import argparse
import os
import pickle
import struct

INDEX_SUBPATH: str = "index.bin"
VOCABULARY_SUBPATH: str = "vocabulary.pkl"
DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
TERM_ID_SUBPATH: str = "term_index.pkl"
SKIPLIST_SUBPATH: str = "skiplist.bin"
SKIP_INDEX_SUBPATH: str = "skip_index.pkl"


def retrieve_skip(index_path, term, skip_index, term_to_id):
    if term not in term_to_id:
        return {}
    term_id = term_to_id[term]

    if term_id not in skip_index:
        return {}

    seek, amount = skip_index[term_id]
    skips = {}

    with open(os.path.join(index_path, SKIPLIST_SUBPATH), 'rb') as f:
        f.seek(seek)

        for _ in range(amount):
            term_id_read, src, dst = struct.unpack('>III', f.read(12))
            skips[src] = dst

    return skips

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Directory with the index files")
    parser.add_argument("term", help="Term to search for")

    args = parser.parse_args()

    with open(os.path.join(args.index_path, SKIP_INDEX_SUBPATH), 'rb') as f:
        skip_index = pickle.load(f)

    with open(os.path.join(args.index_path, TERM_ID_SUBPATH), 'rb') as f:
        term_to_id = pickle.load(f)

    with open(os.path.join(args.index_path, DOCUMENT_INDEX_SUBPATH), 'rb') as f:
        doc_index = pickle.load(f)

    skips = retrieve_skip(args.index_path, args.term, skip_index, term_to_id)
    
    print(f"Skips for term '{args.term}':")
    for src, dst in skips.items():
        print(f"  Skip from position {src} to position {dst}")

if __name__ == "__main__":
    main()