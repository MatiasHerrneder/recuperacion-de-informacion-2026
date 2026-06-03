import argparse
import pickle
import os
from PostingList import InMemoryPosting as PostingList
import re
from retrieval import retrieve_posting


DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"

def retrieve_TAAT(index_path, query):
    VOCABULARY_SUBPATH: str = "vocabulary.pkl"
    DOCUMENT_INDEX_SUBPATH: str = "document_index.pkl"
    TERM_ID_SUBPATH: str = "term_index.pkl"

    with open(os.path.join(index_path, VOCABULARY_SUBPATH), 'rb') as f:
        vocabulary = pickle.load(f)

    with open(os.path.join(index_path, DOCUMENT_INDEX_SUBPATH), 'rb') as f:
        doc_index = pickle.load(f)

    with open(os.path.join(index_path, TERM_ID_SUBPATH), 'rb') as f:
        term_to_id = pickle.load(f)

    universe = PostingList(sorted(doc_index.keys()), [1.0] * len(doc_index))  # pesos dummy
    
    return evaluate(to_postfix(tokenize(query)), index_path, vocabulary, term_to_id, universe)


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

def evaluate(tokens, index_path, vocabulary, term_to_id, universe):
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
            doc_ids, freqs = retrieve_posting(index_path, token, vocabulary, term_to_id)
            stack.append(PostingList(doc_ids, freqs))
    return stack.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Directory with the index files")
    parser.add_argument("query", help="Query to search for")
    args = parser.parse_args()

    document_index = pickle.load(open(os.path.join(args.index_path, DOCUMENT_INDEX_SUBPATH), 'rb'))
    print(f"Document index: {document_index} documents")

    result = retrieve_TAAT(args.index_path, args.query)
    for doc_id in result.docids:
        # print(doc_index[doc_id])
        print(f"{document_index[doc_id]}:{doc_id}")


if __name__ == "__main__":
    main()
