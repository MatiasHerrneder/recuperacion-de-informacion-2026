import os
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