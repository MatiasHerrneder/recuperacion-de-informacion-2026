import argparse
import math
import pickle
import os
from PostingList import InMemoryPosting as PostingList
from retrieval import retrieve_posting
from tokenizer import tokenizer
import heapq


DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
VOCABULARY_SUBPATH: str = "vocabulary.pkl"
DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
TERM_ID_SUBPATH: str = "term_index.pkl"
IDFS_SUBPATH: str = "idf.pkl"
DOC_NORMS_SUBPATH: str = "doc_norms.pkl"
STOPWORDS_SUBPATH: str = "stop_words.pkl"

MIN_TERM_LENGTH: int = 3
MAX_TERM_LENGTH: int = 100

def retrieve_TAAT(index_path, query, k: int = 10):
    
    stop_words_path = os.path.join(index_path, STOPWORDS_SUBPATH)
    if os.path.exists(stop_words_path):
        with open(stop_words_path, 'rb') as f:
            stop_words = pickle.load(f)
    else:
        stop_words = set()

    with open(os.path.join(index_path, VOCABULARY_SUBPATH), 'rb') as f:
        vocabulary = pickle.load(f)

    # with open(os.path.join(index_path, DOCUMENT_INDEX_SUBPATH), 'rb') as f:
    #     doc_index = pickle.load(f)

    with open(os.path.join(index_path, TERM_ID_SUBPATH), 'rb') as f:
        term_to_id = pickle.load(f)

    with open(os.path.join(index_path, IDFS_SUBPATH), 'rb') as f:
        idfs = pickle.load(f)

    with open(os.path.join(index_path, DOC_NORMS_SUBPATH), 'rb') as f:
        doc_norms = pickle.load(f)
    
    # tf-idf de la query
    query_terms = tokenizer(query)
    query_tf = {}
    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1

    scores = {} # numerador del coseno

    for term, tf_q in query_tf.items():
        if term not in term_to_id:
            continue
        term_id = term_to_id[term]
        if term_id not in idfs:
            continue

        tfidf_q = tf_q * idfs[term_id]

        # recorrer posting del termino
        doc_ids, freqs = retrieve_posting(index_path, term, vocabulary, term_to_id)
        for doc_id, tf_d in zip(doc_ids, freqs):
            tfidf_d = tf_d * idfs[term_id]
            scores[doc_id] = scores.get(doc_id, 0) + tfidf_q * tfidf_d

    # normalizar por normas
    query_norm = math.sqrt(sum((tf * idfs[term_to_id[t]]) ** 2
                    for t, tf in query_tf.items()
                    if t in term_to_id))

    if query_norm == 0:
        return []

    heap = []
    for doc_id, score in scores.items():
        cos = score / (query_norm * doc_norms.get(doc_id, 1.0))
        if len(heap) < k:
            heapq.heappush(heap, (cos, doc_id))
        elif cos > heap[0][0]:
            heapq.heapreplace(heap, (cos, doc_id))

    return sorted(heap, reverse=True)

def retrieve_DAAT(index_path, query, k=10):

    stop_words_path = os.path.join(index_path, STOPWORDS_SUBPATH)
    if os.path.exists(stop_words_path):
        with open(stop_words_path, 'rb') as f:
            stop_words = pickle.load(f)
    else:
        stop_words = set()

    with open(os.path.join(index_path, VOCABULARY_SUBPATH), 'rb') as f:
        vocabulary = pickle.load(f)

    # with open(os.path.join(index_path, DOCUMENT_INDEX_SUBPATH), 'rb') as f:
    #     doc_index = pickle.load(f)

    with open(os.path.join(index_path, TERM_ID_SUBPATH), 'rb') as f:
        term_to_id = pickle.load(f)

    with open(os.path.join(index_path, IDFS_SUBPATH), 'rb') as f:
        idfs = pickle.load(f)

    with open(os.path.join(index_path, DOC_NORMS_SUBPATH), 'rb') as f:
        doc_norms = pickle.load(f)

    query_terms = tokenizer(query)

    query_tf = {}
    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1

    postings = []

    for term, tf_q in query_tf.items():

        if term not in term_to_id:
            continue

        term_id = term_to_id[term]

        doc_ids, freqs = retrieve_posting(
            index_path,
            term,
            vocabulary,
            term_to_id
        )

        postings.append({
            "doc_ids": doc_ids,
            "freqs": freqs,
            "pos": 0,
            "tfidf_q": tf_q * idfs[term_id],
            "idf": idfs[term_id]
        })

    scores = {}

    while True:

        active_docs = [
            p["doc_ids"][p["pos"]]
            for p in postings
            if p["pos"] < len(p["doc_ids"])
        ]

        if not active_docs:
            break

        current_doc = min(active_docs)

        score = 0

        for p in postings:

            pos = p["pos"]

            if pos >= len(p["doc_ids"]):
                continue

            if p["doc_ids"][pos] == current_doc:

                tfidf_d = p["freqs"][pos] * p["idf"]

                score += p["tfidf_q"] * tfidf_d

                p["pos"] += 1

        scores[current_doc] = score

    query_norm = math.sqrt(
        sum(
            (tf * idfs[term_to_id[t]]) ** 2
            for t, tf in query_tf.items()
            if t in term_to_id
        )
    )

    if query_norm == 0:
        return []

    heap = []

    for doc_id, score in scores.items():

        cos = score / (
            query_norm *
            doc_norms.get(doc_id, 1.0)
        )

        if len(heap) < k:
            heapq.heappush(heap, (cos, doc_id))
        elif cos > heap[0][0]:
            heapq.heapreplace(heap, (cos, doc_id))

    return sorted(heap, reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Base directory with the index files")
    parser.add_argument("query", help="Query to search for")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    document_index = pickle.load(open(os.path.join(args.index_path, DOCUMENT_INDEX_SUBPATH), 'rb'))

    result = retrieve_DAAT(args.index_path, args.query, k=args.k)
    for score, doc_id in result:
        print(f"{document_index[doc_id]}:{doc_id}:{score}")


if __name__ == "__main__":
    main()
