import argparse
import pickle
import os
import struct
from PostingList import DiskPostingList
import re


INDEX_SUBPATH: str = "index.bin"
VOCABULARY_SUBPATH: str = "vocabulary.pkl"
DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
TERM_ID_SUBPATH: str = "term_index.pkl"
SKIPLIST_SUBPATH: str = "skiplist.bin"
SKIP_INDEX_SUBPATH: str = "skip_index.pkl"


def load_skips(index_path, term_id, skip_index):
    skips = {}
    if term_id not in skip_index:
        return skips
    skip_seek, amount = skip_index[term_id]
    with open(os.path.join(index_path, SKIPLIST_SUBPATH), 'rb') as f:
        f.seek(skip_seek)
        for _ in range(amount):
            _, src, dst = struct.unpack('>III', f.read(12))
            skips[src] = dst
    return skips

def retrieve_TAAT(index_path, query):

    with open(os.path.join(index_path, VOCABULARY_SUBPATH), 'rb') as f:
        vocabulary = pickle.load(f)

    with open(os.path.join(index_path, DOCUMENT_INDEX_SUBPATH), 'rb') as f:
        doc_index = pickle.load(f)

    with open(os.path.join(index_path, TERM_ID_SUBPATH), 'rb') as f:
        term_to_id = pickle.load(f)

    with open(os.path.join(index_path, SKIP_INDEX_SUBPATH), 'rb') as f:
        skip_index = pickle.load(f)

    from PostingList import InMemoryPosting
    universe = InMemoryPosting(sorted(doc_index.keys()), [1.0] * len(doc_index))
    
    return evaluate(to_postfix(tokenize(query)), index_path, vocabulary, term_to_id, skip_index, universe)


def tokenize(query: str) -> list[str]:
    return re.findall(r'\(|\)|AND|OR|NOT|\w+', query)

PRECEDENCE = {'NOT': 3, 'AND': 2, 'OR': 1}

def to_postfix(tokens: list[str]) -> list[str]:
    output = []
    operators = []

    for token in tokens:
        if token == '(':
            operators.append(token)

        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()

        elif token in PRECEDENCE:
            while (operators
                   and operators[-1] != '('
                   and operators[-1] in PRECEDENCE
                   and PRECEDENCE[operators[-1]] >= PRECEDENCE[token]):
                output.append(operators.pop())
            operators.append(token)

        else:
            output.append(token)

    while operators:
        output.append(operators.pop())

    return output

def evaluate(tokens, index_path, vocabulary, term_to_id, skip_index, universe):
    stack = []
    for token in tokens:
        if token == 'NOT':
            a = stack.pop()
            stack.append(a.posting_not(universe))
        elif token == 'AND':
            b, a = stack.pop(), stack.pop()
            stack.append(a.posting_and(b))
        elif token == 'OR':
            b, a = stack.pop(), stack.pop()
            stack.append(a.posting_or(b))
        else:
            if token not in term_to_id or term_to_id[token] not in vocabulary:
                stack.append(DiskPostingList(
                    index_file=os.path.join(index_path, INDEX_SUBPATH),
                    seek=0, length=0, skips={}
                ))
                continue
            term_id = term_to_id[token]
            seek, length = vocabulary[term_id]
            skips = load_skips(index_path, term_id, skip_index)
            stack.append(DiskPostingList(
                index_file=os.path.join(index_path, INDEX_SUBPATH),
                seek=seek,
                length=length,
                skips=skips
            ))
    return stack.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Directory with the index files")
    parser.add_argument("query", help="Query to search for")
    args = parser.parse_args()

    document_index = pickle.load(open(os.path.join(args.index_path, DOCUMENT_INDEX_SUBPATH), 'rb'))
    # print(f"Document index: {document_index} documents")

    result = retrieve_TAAT(args.index_path, args.query)
    results = []

    if hasattr(result, 'docids'):  # InMemoryPosting
        results = result.docids
    else:  # DiskPostingList
        result.reset()
        while result.docid() is not None:
            results.append(result.docid())
            result.next()

    for doc_id in results:
        print(f"{document_index[doc_id]}:{doc_id}")


if __name__ == "__main__":
    main()
